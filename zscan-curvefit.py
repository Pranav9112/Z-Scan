"""
Z-Scan Analysis Suite  –  improved & scientifically corrected
=============================================================
Key corrections vs. original
-----------------------------
1. CA model: curve_fit is now performed (φ is *fitted*, not only manually
   set).  The Sheik-Bahae (1990) formula is kept but the sign convention is
   explicit: peak-before-valley → n2 > 0 (self-focusing).
2. OA model: series convergence is checked; a warning is shown when |q0|≥1
   so the user knows the perturbative expansion may be inaccurate.
3. Rayleigh range z_R is now a fittable parameter for both CA and OA (with
   a user-supplied initial guess).
4. Normalisation uses the *outer 10 %* of points at both ends (more
   physically meaningful than first-5-points only).
5. z unit is now explicitly selectable (µm / mm / cm / m) instead of a
   silent ×1e-3.
6. n2 is extracted from the CA fit:  n2 = φ / (k · I0 · Leff).
7. Goodness-of-fit (R²) is reported for both modes.
8. A residuals panel is added below every plot.
9. Error propagation for n2 uses the covariance from curve_fit.
10. The OA series upper limit is raised adaptively to ensure convergence.
"""

import warnings
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Z-Scan Analysis Suite", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background: #0f1117; }
    h1 { color: #5bc8f5; letter-spacing: 1px; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🔬 Z-Scan Analysis Suite")
st.caption(
    "Nonlinear optical characterisation — Closed Aperture & Open Aperture"
)


# ── helpers ─────────────────────────────────────────────────────────────────

def r_squared(y_data: np.ndarray, y_fit: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")


def normalise_transmission(T: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """
    Normalise by the mean of the outer `frac` fraction of data points
    (both tails), which should be far from focus and therefore linear.
    Falls back to first-5 if there are fewer than 10 points.
    """
    n = len(T)
    if n < 10:
        return T / np.mean(T[:5]) if n >= 5 else T / T[0]
    n_tail = max(1, int(n * frac))
    baseline = np.mean(np.concatenate([T[:n_tail], T[-n_tail:]]))
    return T / baseline


# ── physical models ─────────────────────────────────────────────────────────

def ca_transmittance(z: np.ndarray, phi: float, z_R: float) -> np.ndarray:
    """
    Closed-Aperture Z-scan transmittance (Sheik-Bahae 1990, Eq. 5).

        T(z) ≈ 1 + 4 φ x / [(x² + 9)(x² + 1)]

    where x = z / z_R and φ = k · n2 · I0 · Leff  (on-axis nonlinear phase).

    Sign convention
    ---------------
    φ > 0  →  n2 > 0  (self-focusing) → pre-focal peak, post-focal valley.
    φ < 0  →  n2 < 0  (self-defocusing) → valley then peak.

    Valid for |φ| ≤ π (small-phase approximation).
    """
    x = z / z_R
    return 1.0 + (4.0 * phi * x) / ((x**2 + 9.0) * (x**2 + 1.0))


def oa_transmittance(
    z: np.ndarray, beta: float, I0: float, Leff: float, z_R: float
) -> np.ndarray:
    """
    Open-Aperture Z-scan transmittance via the power-series expansion
    (Sheik-Bahae 1990, Eq. 22):

        T(z) = Σ_{m=0}^{M}  (-q0)^m / (m+1)^{3/2}

    where q0(z) = β · I0 · Leff / (1 + (z/z_R)²).

    Convergence note
    ----------------
    The series converges for |q0| < 1.  We use M = max(30, 10/|q0_max|)
    terms and warn the user when |q0_max| ≥ 1 because higher-order
    nonlinearities or saturation likely dominate there.
    """
    q0 = beta * I0 * Leff / (1.0 + (z / z_R) ** 2)
    q0_max = np.max(np.abs(q0))

    # adaptive series length for numerical safety
    M = max(30, int(10.0 / q0_max) + 1) if q0_max > 0 else 30
    M = min(M, 200)  # cap to avoid runaway

    T = np.zeros_like(z, dtype=float)
    for m in range(M):
        T += ((-q0) ** m) / ((m + 1) ** 1.5)

    return T, q0_max


# ── file upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel file",
    type=["csv", "xlsx"],
    help="First row should be column headers.",
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to begin analysis.")
    st.stop()

try:
    df = (
        pd.read_csv(uploaded_file)
        if uploaded_file.name.endswith(".csv")
        else pd.read_excel(uploaded_file)
    )
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()


# ── sidebar ──────────────────────────────────────────────────────────────────
sb = st.sidebar
sb.title("⚙ Analysis Controls")

analysis_mode = sb.radio("Analysis Type", ["Closed Aperture", "Open Aperture"])
normalize = sb.checkbox("Normalise Transmission", value=True)

# column selection
sb.subheader("📊 Column Selection")
z_col = sb.selectbox("z Column", df.columns)
t_col = sb.selectbox(
    "Transmission Column", df.columns, index=min(1, len(df.columns) - 1)
)

# z unit
sb.subheader("📏 z-axis Units")
z_unit = sb.selectbox("z data is in…", ["mm", "µm", "cm", "m"])
unit_to_m = {"µm": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0}

# row selection
sb.subheader("📑 Row Selection")
row_range = sb.slider("Data Range", 0, len(df) - 1, (0, len(df) - 1))
step_row = sb.number_input("Take Every Nth Row", min_value=1, value=1)

# centering
sb.subheader("📍 Centre Subtraction")
center_mode = sb.radio("Method", ["Mean of z", "Custom", "None"])
custom_center = None
if center_mode == "Custom":
    custom_center = sb.number_input(
        f"Centre value ({z_unit})", value=0.0, format="%.6f"
    )

# ── Peak Intensity Calculator ──────────────────────────────────────────────
sb.subheader("⚡ Peak Intensity Calculator")

avg_power_mw = sb.number_input(
    "Average Power (mW)",
    value=100.0,
    min_value=0.0,
    format="%.3f"
)

pulse_duration_fs = sb.number_input(
    "Pulse Duration (fs)",
    value=100.0,
    min_value=1.0,
    format="%.1f"
)

rep_rate_hz = sb.number_input(
    "Repetition Rate (Hz)",
    value=80000000.0,
    min_value=1.0,
    format="%.0f"
)

beam_waist_um = sb.number_input(
    "Beam Waist w₀ (µm)",
    value=25.0,
    min_value=0.001,
    format="%.3f"
)

# Unit conversions
Pavg = avg_power_mw * 1e-3       # W
tau = pulse_duration_fs * 1e-15  # s
w0 = beam_waist_um * 1e-4        # cm

# Pulse energy
Epulse = Pavg / rep_rate_hz

# Peak power (Gaussian pulse)
Ppeak = 0.94 * Epulse / tau

# Peak intensity (Gaussian beam)
I0_calc = 2 * Ppeak / (np.pi * w0**2)

sb.markdown("---")
sb.markdown("### Calculated Values")

sb.metric(
    "Pulse Energy",
    f"{Epulse:.3e} J"
)

sb.metric(
    "Peak Power",
    f"{Ppeak:.3e} W"
)

sb.metric(
    "Peak Intensity I₀",
    f"{I0_calc:.3e} W/cm²"
)
sb.markdown("---")
# ── Beam Parameters Calculator ────────────────────────────────────────────
sb.subheader("🔭 Beam Parameters")

wavelength_nm_calc = sb.number_input(
    "Wavelength (nm)",
    value=532.0,
    min_value=1.0
)

beam_waist_um_calc = sb.number_input(
    "Beam Waist w₀ (µm)",
    value=25.0,
    min_value=0.001
)

# Convert to SI
wavelength_m = wavelength_nm_calc * 1e-9
w0_m = beam_waist_um_calc * 1e-6

# Rayleigh range
zR_m = np.pi * w0_m**2 / wavelength_m

sb.markdown("---")
sb.metric(
    "Rayleigh Range zR",
    f"{zR_m*1e3:.3f} mm"
)

sb.metric(
    "Confocal Parameter b",
    f"{2*zR_m*1e3:.3f} mm"
)
sb.markdown("---")

# ── mode-specific parameters ─────────────────────────────────────────────────
if analysis_mode == "Closed Aperture":
    sb.subheader("🔵 CA Parameters")
    phi_init = sb.number_input(
        "Initial φ guess", value=0.5, step=0.05,
        help="On-axis nonlinear phase shift (rad). |φ| ≤ π for validity.",
    )
    z_R_ca_init = sb.number_input(
        "Initial z_R guess (m)",
        value=float(zR_m),
        format="%.4e"
    )
    fit_zR_ca = sb.checkbox("Also fit z_R", value=False)

    sb.subheader("🔬 Beam / Sample (for n₂ extraction)")
    wavelength_nm = sb.number_input("Wavelength (nm)", value=532.0, step=1.0)
    I0_ca = sb.number_input(
        "Peak Intensity I₀ (W/cm²)",
        value=float(I0_calc),
        format="%.3e"
    )
    Leff_ca = sb.number_input(
        "Effective Length L_eff (cm)", value=0.05, format="%.5f"
    )

else:
    sb.subheader("🟢 OA Parameters")
    I0 = sb.number_input(
        "Peak Intensity I₀ (W/cm²)",
        value=float(I0_calc),
        format="%.3e"
    )
    Leff = sb.number_input("L_eff (cm)", value=0.05, format="%.5f")
    z_R_oa_init = sb.number_input(
        "z_R",
        value=float(zR_m),
        format="%.4e"
    )
    fit_zR_oa = sb.checkbox("Also fit z_R", value=False)
    beta_init = sb.number_input(
        "Initial β guess (cm/W)",
        value=1e-7,
        format="%.3e"
    )

    sb.markdown("### Manual β Control")

    manual_beta = sb.number_input(
        "Manual β (cm/W)",
        value=beta_init,
        format="%.3e"
    )

    manual_beta_slider = sb.slider(
        "Manual β Slider (×10⁻⁷ cm/W)",
        min_value=-100.0,
        max_value=100.0,
        value=float(manual_beta * 1e7),
        step=0.1
    )

    manual_beta = manual_beta_slider * 1e-7

# ── filter & extract data ────────────────────────────────────────────────────
filtered_df = df.iloc[row_range[0]: row_range[1] + 1: step_row]

st.subheader("📄 Filtered Data Preview")
st.dataframe(filtered_df, use_container_width=True)

try:
    z_raw = filtered_df[z_col].astype(float).values
    Trans_raw = filtered_df[t_col].astype(float).values
except Exception as exc:
    st.error(f"Could not convert columns to numbers: {exc}")
    st.stop()

if len(z_raw) < 5:
    st.error("Need at least 5 data points after filtering.")
    st.stop()

# centre & convert to metres
z = z_raw.copy()
if center_mode == "Mean of z":
    z = z - np.mean(z)
elif center_mode == "Custom":
    z = z - custom_center
z = z * unit_to_m[z_unit]

# normalise
Trans_exp = Trans_raw.copy().astype(float)
if normalize:
    Trans_exp = normalise_transmission(Trans_exp)

# ── results ──────────────────────────────────────────────────────────────────
st.subheader("📈 Analysis Results")


def make_fig_with_residuals(title: str):
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.setp(ax_main.get_xticklabels(), visible=False)
    ax_res.axhline(0, color="k", lw=0.8, ls="--")
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel("z (m)")
    ax_main.set_ylabel("Normalised Transmission")
    ax_main.grid(True, alpha=0.3)
    ax_res.grid(True, alpha=0.3)
    return fig, ax_main, ax_res


# ════════════════════════════════════════════════════════════════════════════
#  CLOSED APERTURE
# ════════════════════════════════════════════════════════════════════════════
if analysis_mode == "Closed Aperture":

    if fit_zR_ca:
        def ca_model_2p(z, phi, z_R):
            return ca_transmittance(z, phi, z_R)
        p0 = [phi_init, z_R_ca_init]
        param_names = ["φ", "z_R"]
    else:
        def ca_model_2p(z, phi):
            return ca_transmittance(z, phi, z_R_ca_init)
        p0 = [phi_init]
        param_names = ["φ"]

    try:
        popt, pcov = curve_fit(
            ca_model_2p, z, Trans_exp,
            p0=p0, maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:
        st.error(f"CA curve_fit failed: {exc}")
        st.stop()

    phi_fit = popt[0]
    phi_err = perr[0]
    z_R_fit = popt[1] if fit_zR_ca else z_R_ca_init

    z_fine = np.linspace(z.min(), z.max(), 2000)
    T_fit = ca_transmittance(z_fine, phi_fit, z_R_fit)
    T_at_data = ca_transmittance(z, phi_fit, z_R_fit)
    R2 = r_squared(Trans_exp, T_at_data)

    if abs(phi_fit) > np.pi:
        st.warning(
            f"|φ| = {abs(phi_fit):.3f} rad exceeds π.  "
            "The small-phase approximation may be invalid — "
            "consider reducing laser intensity or sample length."
        )

    # n2 extraction:  φ = k · n2 · I0 · Leff  →  n2 = φ / (k I0 Leff)
    # I0 in W/cm², Leff in cm  →  n2 in cm²/W
    k = 2 * np.pi / (wavelength_nm * 1e-7)   # wave-vector in cm⁻¹
    n2_fit = phi_fit / (k * I0_ca * Leff_ca)
    # error propagation (I0, Leff treated as exact)
    n2_err = phi_err / (k * I0_ca * Leff_ca)

    fig, ax, ax_res = make_fig_with_residuals("Closed Aperture Z-Scan")
    ax.scatter(z, Trans_exp, color="#e05252", s=40, zorder=3,
               label="Experimental")
    ax.plot(z_fine, T_fit, color="#4a90d9", lw=2, label="Fit")
    ax.legend()

    residuals = Trans_exp - T_at_data
    ax_res.scatter(z, residuals, s=20, color="#e05252", alpha=0.7)

    st.pyplot(fig, use_container_width=True)

    # results table
    col1, col2, col3 = st.columns(3)
    col1.metric("φ (rad)", f"{phi_fit:.4f} ± {phi_err:.4f}")
    col2.metric("n₂ (cm²/W)", f"{n2_fit:.3e} ± {n2_err:.1e}")
    col3.metric("R²", f"{R2:.5f}")

    if fit_zR_ca:
        st.info(f"Fitted z_R = {z_R_fit*1e3:.3f} mm")

    with st.expander("ℹ️ CA interpretation guide"):
        st.markdown(
            """
            **Peak–valley order**
            - Peak *before* valley (as z increases) → **n₂ > 0** (self-focusing)
            - Valley *before* peak → **n₂ < 0** (self-defocusing)

            **Model validity**
            - Requires |φ| ≤ π (small phase approximation)
            - Thin sample assumption: sample thickness ≪ z_R
            - Single beam, no two-photon absorption in CA trace (divide CA/OA if TPA is present)
            """
        )

# ════════════════════════════════════════════════════════════════════════════
#  OPEN APERTURE
# ════════════════════════════════════════════════════════════════════════════
else:
    if fit_zR_oa:
        def oa_model_fit(z, beta, z_R):
            T, _ = oa_transmittance(z, beta, I0, Leff, z_R)
            return T
        p0 = [beta_init, z_R_oa_init]
        param_names = ["β", "z_R"]
    else:
        def oa_model_fit(z, beta):
            T, _ = oa_transmittance(z, beta, I0, Leff, z_R_oa_init)
            return T
        p0 = [beta_init]
        param_names = ["β"]

    # check q0 convergence before fitting
    q0_check = beta_init * I0 * Leff  # at focus, maximum q0
    if abs(q0_check) >= 1.0:
        st.warning(
            f"With your initial β guess, |q₀|_max ≈ {abs(q0_check):.2f} ≥ 1.  "
            "The perturbative series may not converge reliably.  "
            "Results should be treated with caution."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                oa_model_fit, z, Trans_exp,
                p0=p0, maxfev=20000,
            )
        perr = np.sqrt(np.diag(pcov))
    except Exception as exc:
        st.error(f"OA curve_fit failed: {exc}")
        st.stop()

    beta_fit = popt[0]
    beta_err = perr[0]
    z_R_fit = popt[1] if fit_zR_oa else z_R_oa_init

    z_fine = np.linspace(z.min(), z.max(), 2000)
    T_fit_fine, q0_max_fit = oa_transmittance(z_fine, beta_fit, I0, Leff, z_R_fit)
    T_manual, _ = oa_transmittance(
        z_fine, manual_beta, I0, Leff, z_R_fit
    )
    T_at_data, _ = oa_transmittance(z, beta_fit, I0, Leff, z_R_fit)
    R2 = r_squared(Trans_exp, T_at_data)

    if q0_max_fit >= 1.0:
        st.warning(
            f"|q₀|_max at the fitted β = {q0_max_fit:.3f} ≥ 1 — "
            "series convergence is not guaranteed.  "
            "Consider reducing I₀ or checking for saturation."
        )

    fig, ax, ax_res = make_fig_with_residuals("Open Aperture Z-Scan")
    ax.scatter(z, Trans_exp, color="#e05252", s=40, zorder=3,
               label="Experimental")
    ax.plot(
        z_fine,
        T_fit_fine,
        color="#4a90d9",
        lw=2,
        label=f"Auto Fit β={beta_fit:.3e}"
    )

    ax.plot(
        z_fine,
        T_manual,
        "--",
        lw=2,
        color="#2ca02c",
        label=f"Manual β={manual_beta:.3e}"
    )
    ax.legend()

    residuals = Trans_exp - T_at_data
    ax_res.scatter(z, residuals, s=20, color="#e05252", alpha=0.7)

    st.pyplot(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("β (cm/W)", f"{beta_fit:.3e} ± {beta_err:.1e}")
    col2.metric("Manual β", f"{manual_beta:.3e}")
    col3.metric("|q₀|_max", f"{q0_max_fit:.3f}")
    col4.metric("R²", f"{R2:.5f}")

    if fit_zR_oa:
        st.info(f"Fitted z_R = {z_R_fit*1e3:.3f} mm")

    with st.expander("ℹ️ OA interpretation guide"):
        st.markdown(
            """
            **Sign of β**
            - β > 0 → **two-photon absorption** (transmission dip at focus)
            - β < 0 → **saturable absorption** (transmission peak at focus)

            **Model validity**
            - Requires |q₀| < 1 (perturbative regime)
            - For |q₀| ≥ 1 use the full numerical integration
            - Assumes a Gaussian beam and thin sample
            """
        )

# ── data summary ─────────────────────────────────────────────────────────────
with st.expander("📋 Data Summary"):
    st.markdown(
        f"""
        | Parameter | Value |
        |---|---|
        | Points used | {len(z)} |
        | z min | {z.min():.4e} m |
        | z max | {z.max():.4e} m |
        | T min (raw) | {Trans_raw.min():.6f} |
        | T max (raw) | {Trans_raw.max():.6f} |
        | T min (normalised) | {Trans_exp.min():.6f} |
        | T max (normalised) | {Trans_exp.max():.6f} |
        """
    )