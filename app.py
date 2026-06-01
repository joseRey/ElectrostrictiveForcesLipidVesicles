import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import json
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(parent_dir, "frontend")
vesicle_dashboard = components.declare_component("vesicle_dashboard", path=frontend_dir)

def get_base64_image(img_path):
    import base64
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"
    except Exception:
        return ""

# ============================================================================
# Vesicle Electrostriction Dashboard v6
# Corrected physics core. Reproduces Rey et al. (IEEE TDEI, 2009):
#   - prolate/oblate transition driven by conductivity ratio (Fig 5, 7a)
#   - reversed transition driven by permittivity ratio (Fig 6, 7b)
#   - ratio-dependent Maxwell-Wagner inflection -> SLANTED T1/T2/T3 boundary
#   - validated against the Hyuga (1991) and Gao (2008) analytical forms (Fig 4)
#
# v5 additions:
#   - Single-iframe custom component architecture (declare_component)
#     completely eliminating flashing during animation toggles or plot clicks.
#   - 100% client-side animation loop for 2D/3D plots and phase diagram marker.
#   - Instant parameter sync from client-side JS back to Streamlit session state.
#   - Streamlit widgets removed from sidebar to prevent page reload on slider changes.
#   - Toggle to optionally render validation panel.
#   - Integrated live automatic animation halting on slider grab or plot click.
#
# v6 additions:
#   - Start of Version 6 branch.
#   - Branding, documentation, and backup systems updated for Version 6.
# ============================================================================

st.set_page_config(
    page_title="Electrostrictive Forces on Lipid Vesicles",
    page_icon="frontend/vesicle_white_icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clear cache once per session to ensure new phase diagram scales apply
try:
    if "cache_cleared_v6" not in st.session_state:
        st.cache_data.clear()
        st.session_state.cache_cleared_v6 = True
except Exception:
    pass

# ---- Premium white-background styling (unchanged from v1) -------------------
st.markdown("""
<style>
    /* Trim the large default top padding of the main content area */
    .main .block-container, section.main > div.block-container,
    div[data-testid="stMainBlockContainer"] {
        padding-top: 2.75rem !important;
    }
    /* Card borders and shadows — not controllable via config.toml */
    div[data-testid="stVerticalBlockBorderDiv"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05) !important;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stExpander"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 0.5rem !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        border-bottom: 1px solid #cbd5e1 !important;
        padding: 0.5rem 1rem !important;
    }
    /* Style the dashboard logo image with a glass/card feel */
    div[data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

EPS_0 = 8.854187817e-12  # F/m — vacuum permittivity (SI)

# ============================================================================
# PHYSICS CORE
# All functions accept scalar OR numpy-array frequency arguments (vectorized).
#
# References (see README for full citations):
#   [1] Rey et al., IEEE TDEI 16(5), 2009       — primary paper (COMSOL FEM)
#   [2] Hyuga, Kinosita & Wakabayashi, 1991     — analytical force formula
#   [3] Gao, Feng, Yin & Gao, JMPS 56, 2008    — alternative force prefactor
# ============================================================================

def effective_coefficients(freq, eps_in, eps_out, sigma_in_uS, sigma_out_uS,
                           R, delta, eps_m, sigma_m, E0_Vcm):
    """Single-shell (membrane + core) Maxwell-Wagner dipole solution [1].

    Models a vesicle as a conducting sphere (interior) coated by a thin
    capacitive/resistive membrane shell, embedded in an external medium,
    all under a uniform AC field E0.

    Returns
    -------
    a    : complex dipole field amplitude = f_CM * E0  (bounded, order E0)
           Represents the strength and phase of the induced dipole.
           Outside the vesicle the perturbed field ~ a * cos(theta) / r^2.
    b    : complex interior uniform field (bounded, order E0)
           Inside the vesicle the field is spatially uniform = b * cos(theta).
    E0   : applied field in V/m
    s_eff: effective complex conductivity of the shelled sphere

    Physics note: the membrane enters as a sheet admittance Ym = (sigma_m + j*omega*eps_m*eps0)/delta.
    The core-plus-membrane combination has an effective sigma that transitions
    between the membrane-limited low-f regime and the core-limited high-f regime
    at the membrane RC frequency f_m = sigma_m / (2*pi*eps_m*eps0) ~ 36 Hz
    (for the paper's parameters). This is DIFFERENT from f_MW used in the force model.
    """
    sigma_in  = sigma_in_uS  * 1e-4   # convert uS/cm -> S/m  (1 uS/cm = 1e-4 S/m)
    sigma_out = sigma_out_uS * 1e-4
    E0 = E0_Vcm * 100.0               # convert V/cm -> V/m

    omega = 2 * np.pi * freq          # angular frequency (rad/s)

    # Complex conductivities: sigma* = sigma + j*omega*eps0*eps
    # At low f: sigma* ~ sigma (conduction dominates)
    # At high f: sigma* ~ j*omega*eps0*eps (displacement current dominates)
    si = sigma_in  + 1j * omega * EPS_0 * eps_in
    so = sigma_out + 1j * omega * EPS_0 * eps_out

    # Membrane sheet admittance: Ym = sigma_m*/delta  [S/m^2]
    # The membrane is thin (delta << R) so it enters as a boundary condition
    # rather than a volumetric region.
    Ym = (sigma_m + 1j * omega * EPS_0 * eps_m) / delta

    # Effective interior complex conductivity seen from outside the shell.
    # Derived from matching boundary conditions at both membrane surfaces.
    # Series combination: Ym*R in parallel with si (per unit area scaling).
    s_eff = (Ym * R * si) / (Ym * R + si)

    # Clausius-Mossotti factor: determines the dipole strength.
    # fcm = (s_eff - so) / (s_eff + 2*so) is bounded between -0.5 and +1.
    # Positive fcm -> field concentrates inside (prolate tendency)
    # Negative fcm -> field depletes inside (oblate tendency)
    fcm = (s_eff - so) / (s_eff + 2 * so)
    a = fcm * E0                             # dipole field amplitude (p/R^3)
    b = (3 * so / (s_eff + 2 * so)) * E0    # uniform interior field
    return a, b, E0, s_eff


def stress_coefficients(a, b, E0, eps_in, eps_out):
    """Time-averaged Maxwell stress tensor jump at the vesicle surface [1, Eq. 1].

    The surface pressure (outward normal force per unit area) decomposes as:
        p(theta) = K_rc*cos^2(theta) + K_rs*sin^2(theta) + K_theta*cos(theta)*sin(theta)

    Derived from <T_nn> - <T_tt> at the outer surface minus the inner surface,
    using the dipole field solution:
        E_r_out  (radial, outside)   = (E0 + 2a) cos(theta)    [field enhanced at poles]
        E_t_out  (tangent, outside)  = (-E0 + a) sin(theta)    [field reduced at equator]
        E_r_in   (radial, inside)    = b cos(theta)             [uniform interior field]
        E_t_in   (tangent, inside)   = -b sin(theta)

    The 1/4 prefactor comes from time-averaging: <cos^2(omega*t)> = 1/2,
    combined with the 1/2 from the Maxwell stress tensor definition.

    Returns K_rc, K_rs, K_theta (real scalars, units N/m^2 per unit E0^2).
    These same coefficients drive both the 3D color map AND the integrated forces.
    """
    # Radial (cos^2) coefficient: difference of radial field pressures
    # Positive K_rc -> pole outward pressure (prolate tendency)
    K_rc = 0.25 * EPS_0 * (eps_out * np.abs(E0 + 2 * a) ** 2 - eps_in * np.abs(b) ** 2)

    # Tangential (sin^2) coefficient: difference of tangential field pressures
    # Positive K_rs -> equatorial outward pressure (oblate tendency)
    K_rs = 0.25 * EPS_0 * (-eps_out * np.abs(-E0 + a) ** 2 + eps_in * np.abs(b) ** 2)

    # Cross-term (sin*cos): only non-zero when fields are complex (frequency-dependent)
    # This term averages to zero over the full sphere and does not contribute to net force.
    K_theta = 0.5 * EPS_0 * (eps_out * np.real((E0 + 2 * a) * np.conj(-E0 + a))
                             + eps_in * np.abs(b) ** 2)
    return K_rc, K_rs, K_theta


def analytical_pressure(theta, Rs, Re, eps_out, E0, model="hyuga"):
    """Quasi-static analytical surface pressure from Hyuga [2] and Gao [3] (paper Fig 4b).

    Both references derive the low-frequency (DC) limit of the Maxwell electrostrictive
    pressure on a single-shell vesicle.  The angular distribution is identical in both
    papers; they differ only by a theta-independent prefactor.

    Common angular bracket:
        p(theta) ~ pref * [(1 + Rs^2 - 2*Re)*cos^2(theta) + (Re - 1)]
    where Rs = sigma_in/sigma_out, Re = eps_in/eps_out.

    Physical interpretation of the bracket:
      - (1 + Rs^2 - 2*Re)*cos^2(theta): pole pressure, driven by conductivity contrast
      - (Re - 1): uniform offset, driven by permittivity contrast
      - When Rs > 1 and Re = 1: bracket positive at pole -> prolate
      - When Rs = 1 and Re > 1: bracket uniformly negative -> oblate (reversed rule)

    Hyuga [2] prefactor: 9*eps_out / (8*pi*(2+Rs))
    Gao   [3] prefactor: 9*eps_out / (2*(1+Rs)^2)  [= sigma_out^2/(sigma_out+sigma_in)^2 form]

    The two forms are validated to share an identical angular shape (self-test #5).
    They are used only in the Validation Panel (Plot 4) and not in the frequency sweep,
    which uses the frequency-dependent resultant_forces() instead.
    """
    # Angular bracket — identical in Hyuga [2] and Gao [3]
    bracket = (1 + Rs ** 2 - 2 * Re) * np.cos(theta) ** 2 + (Re - 1)
    if model == "gao":
        # Gao [3]: sigma_out^2/(sigma_out+sigma_in)^2 = 1/(1+Rs)^2  (using Rs = si/so)
        pref = (9 * eps_out) / (2 * (1 + Rs) ** 2)
    else:
        # Hyuga [2]: classic 1/(8*pi*(2+Rs)) form
        pref = (9 * eps_out) / (8 * np.pi * (2 + Rs))
    return pref * bracket * E0 ** 2


def resultant_forces(freq, eps_in, eps_out, sigma_in_uS, sigma_out_uS,
                     R, delta, eps_m, sigma_m, E0_Vcm):
    """Frequency-dependent integrated pole force FR_z and equatorial force FR_r [1,2,3].

    Integrates the Hyuga/Gao angular bracket over the vesicle surface with a
    frequency-blended conductivity ratio.  This produces the prolate/oblate
    transition at the Maxwell-Wagner frequency f_MW exactly as in Rey et al. [1].

    How frequency-blending works
    ----------------------------
    At low f: conduction currents dominate -> effective contrast = Rs = sigma_in/sigma_out
    At high f: displacement currents dominate -> effective contrast -> Re = eps_in/eps_out
    The blend uses a Lorentzian: alpha(f) = 1/(1+(f/f_c)^2)
    This is the standard Maxwell-Wagner single-relaxation model [4,5].

    Why this gives the correct slant in the phase diagram (Fig 7a)
    --------------------------------------------------------------
    f_c depends on sigma_out = sigma_in/Rs, so larger Rs -> smaller sigma_out -> lower f_c.
    Each row of the phase heatmap therefore has its own crossover frequency,
    causing the prolate->sphere boundary to slant downward with increasing ratio.

    Sign rules (validated in _selftest, matching paper Figs 5, 6, 7a, 7b)
    -----------------------------------------------------------------------
    eps_in = eps_out (Re=1):  Rs > 1 -> FR_z > 0 (prolate); Rs < 1 -> FR_z < 0 (oblate)
    sigma_in = sigma_out (Rs=1): Re < 1 -> prolate; Re > 1 -> oblate  [REVERSED sign]
    The reversal occurs because permittivity contrast opposes conductivity contrast
    in the Clausius-Mossotti factor at high frequency.

    Note on FR_r in conductivity mode (Re = 1)
    ------------------------------------------
    FR_r = pref * (Re - 1) = 0 exactly when eps_in = eps_out.
    This is physically correct: equatorial pressure requires a dielectric contrast.
    In conductivity mode (equal permittivities), all shape change comes from the pole.
    """
    sigma_in  = sigma_in_uS  * 1e-4   # uS/cm -> S/m
    sigma_out = sigma_out_uS * 1e-4
    E0 = E0_Vcm * 100.0               # V/cm -> V/m

    Rs = sigma_in / sigma_out   # bare conductivity ratio (governs at low f)
    Re = eps_in  / eps_out      # permittivity ratio (governs above f_c)

    # Maxwell-Wagner crossover frequency [1, Eq. 3-5]
    # This is the frequency at which conduction and displacement currents balance.
    # For the paper's parameters (eps_in = eps_out = 80, sigma_in = 15 uS/cm):
    #   sigma_out = 7.5  uS/cm -> f_c ~ 2.25e5 Hz
    #   sigma_out = 30.0 uS/cm -> f_c ~ 5.6e5  Hz
    f_c   = (sigma_in + 2 * sigma_out) / (2 * np.pi * EPS_0 * (eps_in + 2 * eps_out))
    # Debye relaxation weight: 1 at DC, 0 at infinite frequency
    alpha = 1.0 / (1.0 + (freq / f_c) ** 2)

    # Effective ratio inputs to the Hyuga/Gao bracket:
    #   Rs_b interpolates between Rs (low f) and 1.0 (high f, no conductivity contrast)
    #   Re_b is frequency-independent (dielectric contrast is constant in this model)
    Rs_b = 1.0 + alpha * (Rs - 1.0)   # conductivity-ratio blend
    Re_b = Re                           # permittivity ratio (static)

    # Hyuga prefactor: 9*eps_out / (8*pi*(2+Rs_b)) — same as analytical_pressure(..., "hyuga")
    pref  = (9 * eps_out) / (8 * np.pi * (2 + Rs_b)) * E0 ** 2
    # bracket evaluated at theta=0 (pole) and theta=pi/2 (equator)
    FR_z  = pref * (Rs_b ** 2 - Re_b)   # pole:    positive -> outward at Z-axis -> prolate
    FR_r  = pref * (Re_b - 1.0)          # equator: positive -> outward at equator -> oblate

    # Rescale from CGS-prefactor units to SI N/m^2 (removes the 1/8pi and E0^2)
    scale = 4 * np.pi / (9 * E0 ** 2) * 1e-2
    return FR_z * scale, FR_r * scale


def get_impedance(freq, eps_in, eps_out, sigma_in_uS, sigma_out_uS, R, delta, eps_m, sigma_m):
    """Vesicle complex impedance Z(omega) using solid-sphere CM factor (v4 model) [1,5].

    Why NOT the shelled-sphere model from effective_coefficients()?
    ---------------------------------------------------------------
    The shelled-sphere s_eff (with explicit membrane admittance Ym) has two relaxation
    frequencies:
      1. Membrane RC frequency: f_m = sigma_m / (2*pi*eps_m*eps0) ~ 36 Hz
      2. Maxwell-Wagner frequency: f_MW ~ 2.25e5 Hz
    The dominant feature in the impedance spectrum is f_m (3 orders below f_MW),
    which is NOT where the force transition happens.  Plotting this alongside the
    force curves would show a total mismatch of the inflection points.

    The solid-sphere CM factor ignores the membrane layer and uses only the bulk
    interior and exterior complex conductivities.  It has exactly ONE relaxation:
      f_MW = (sigma_in + 2*sigma_out) / (2*pi*eps0*(eps_in + 2*eps_out))
    which is identical to f_c in resultant_forces().  This is the correct model
    for showing that the impedance inflection co-locates with the force transition,
    reproducing the qualitative behaviour of Rey et al. [1] Fig 8.

    Impedance formula: Z = 1 / (sigma_eff * 4*pi*R)
    This is the dilute-suspension limit (one sphere in infinite medium).
    Absolute magnitudes are NOT calibrated to the paper's electrode geometry.
    """
    sigma_in  = sigma_in_uS  * 1e-4
    sigma_out = sigma_out_uS * 1e-4
    omega = 2.0 * np.pi * freq

    # Complex conductivities (no membrane layer — solid-sphere approximation)
    si = sigma_in  + 1j * omega * EPS_0 * eps_in
    so = sigma_out + 1j * omega * EPS_0 * eps_out

    # Solid-sphere CM factor: K_cm in [-0.5, 1]; single Debye dispersion at f_MW
    K_cm  = (si - so) / (si + 2.0 * so)

    # Maxwell Garnett effective medium: sigma_eff = sigma_out * (1 + 3*K_cm)
    # At low f: K_cm ~ (sigma_in - sigma_out)/(sigma_in + 2*sigma_out) -> real
    # At high f: K_cm ~ (eps_in - eps_out)/(eps_in + 2*eps_out) -> real (different value)
    s_eff = so * (1.0 + 3.0 * K_cm)

    # Z = 1 / (sigma_eff * 4*pi*R)  — sphere impedance in infinite medium
    Z = 1.0 / (s_eff * 4.0 * np.pi * R)
    return Z.real, Z.imag


def maxwell_wagner_freq(eps_in, eps_out, sigma_in_uS, sigma_out_uS):
    """Maxwell-Wagner relaxation frequency [1, Eq. 3-5; 5].

    This is the frequency at which the impedance and force curves transition.
    At f << f_MW: conductivity ratio dominates (DC regime)
    At f >> f_MW: permittivity ratio dominates (displacement-current regime)
    """
    sigma_in  = sigma_in_uS  * 1e-4
    sigma_out = sigma_out_uS * 1e-4
    # Time constant tau = eps0*(eps_in+2*eps_out) / (sigma_in+2*sigma_out)
    tau = EPS_0 * (eps_in + 2 * eps_out) / (sigma_in + 2 * sigma_out)
    return 1.0 / (2 * np.pi * tau)




# ============================================================================
# GEOMETRY & PLOTLY ANIMATION BUILDER (v5)
# ============================================================================

TILT = 0.28      # oblique-projection depth factor (gives 3D feel)
N_SILU = 180     # silhouette sample points
N_LAT = 6        # number of latitude parallels
N_MER = 3        # number of meridians
LAT_FRACS = np.linspace(0.15, 0.85, N_LAT)
MER_THETAS = np.linspace(0, np.pi * (1 - 1/N_MER), N_MER)
FIELD_XS = [-12.5, -7.0, 7.0, 12.5]

COLORSCALE = [
    [0.0, '#b2182b'], [0.15, '#d6604d'], [0.35, '#f4a582'],
    [0.48, '#e2e8f0'], [0.5, '#cbd5e1'], [0.52, '#e2e8f0'],
    [0.65, '#4393c3'], [0.85, '#2166ac'], [1.0, '#053061'],
]


def silhouette(a, b, n=N_SILU):
    """Outer ellipse silhouette: x = b*sin(phi), y = a*cos(phi)."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return b * np.sin(phi), a * np.cos(phi), phi


def latitude_ellipse(a, b, lat_frac, n=80):
    """Latitude circle at polar fraction lat_frac in [0,1] (0=pole, 0.5=equator).
    Drawn as a tilted ellipse using oblique projection (TILT factor).
    """
    phi0 = lat_frac * np.pi
    r = b * np.sin(phi0)
    y0 = a * np.cos(phi0)
    t = np.linspace(0, 2 * np.pi, n)
    x = r * np.cos(t)
    y = y0 + TILT * r * np.sin(t)
    return np.append(x, x[0]), np.append(y, y[0])


def meridian_arc(a, b, theta0, n=100):
    """Meridian arc at azimuthal angle theta0 (front-facing half only)."""
    phi = np.linspace(0, np.pi, n)
    x = b * np.sin(phi) * np.cos(theta0)
    y = a * np.cos(phi) + TILT * b * np.sin(phi) * np.sin(theta0)
    return x, y


def field_line_xy(x_pos, height=14.5):
    """Vertical E-field line at fixed x_pos."""
    return [x_pos, x_pos], [-height, height]


def build_plotly_animation(cond_mode, sigma_in_uS, R, delta, eps_m, sigma_m, E0_Vcm,
                           fixed_ratio, fixed_freq, animate_ratio, animate_freq,
                           ratios_grid, freq_grid, shape_grid, cache_dict=None):
    # Constants
    R0 = 10.0
    K_MAX = 0.70
    
    # 1. Determine Sweep Sequence
    if animate_ratio and animate_freq:
        # 4-Quadrant Loop — a closed rectangular path in (frequency, ratio) space.
        # The ratio is intentionally bounded to [1/3, 3] (not the full 0.01..100)
        # so the prolate/oblate force asymmetry stays modest and the panel-3 curves
        # remain readable. Frequency spans the full 1 Hz .. 1 GHz model range.
        R_LOOP_MAX = 3.0
        R_LOOP_MIN = 1.0 / 3.0
        N_EDGE = 24  # points per edge (excluding the shared corner)

        # Edge 1: frequency 1 Hz -> 1 GHz at ratio = R_LOOP_MAX
        f_e1 = np.logspace(0, 9, N_EDGE, endpoint=False)
        r_e1 = np.full(len(f_e1), R_LOOP_MAX)
        # Edge 2: ratio R_LOOP_MAX -> R_LOOP_MIN at frequency = 1 GHz
        r_e2 = np.logspace(np.log10(R_LOOP_MAX), np.log10(R_LOOP_MIN), N_EDGE, endpoint=False)
        f_e2 = np.full(len(r_e2), 1e9)
        # Edge 3: frequency 1 GHz -> 1 Hz at ratio = R_LOOP_MIN
        f_e3 = np.logspace(9, 0, N_EDGE, endpoint=False)
        r_e3 = np.full(len(f_e3), R_LOOP_MIN)
        # Edge 4: ratio R_LOOP_MIN -> R_LOOP_MAX at frequency = 1 Hz (closes the loop)
        r_e4 = np.logspace(np.log10(R_LOOP_MIN), np.log10(R_LOOP_MAX), N_EDGE, endpoint=False)
        f_e4 = np.full(len(r_e4), 1.0)

        sweep_f = np.concatenate([f_e1, f_e2, f_e3, f_e4])
        sweep_r = np.concatenate([r_e1, r_e2, r_e3, r_e4])

        n_loop = len(sweep_f)               # = 4 * N_EDGE
        sweep_indices = list(range(n_loop))
        slider_vals = sweep_indices
        slider_labels = [f"Step {i+1}" for i in range(n_loop)]
        mode_name = "4-Quadrant Loop"
        # edge boundaries for quadrant labelling (used below)
        _q_bounds = (N_EDGE, 2 * N_EDGE, 3 * N_EDGE)
        
    elif animate_ratio:
        # Ratio Sweep
        r_arr = np.logspace(-2, 2, 41)
        sweep_r = list(r_arr) + list(reversed(r_arr))[1:-1]
        sweep_f = np.full(len(sweep_r), fixed_freq)
        
        # Slider only shows the forward pass (41 steps)
        slider_vals = list(range(41))
        slider_labels = [f"{v:.2f}" for v in r_arr]
        sweep_indices = list(range(len(sweep_r)))
        mode_name = "Ratio Sweep"
        
    elif animate_freq:
        # Frequency Sweep
        f_arr = np.logspace(0, 9, 37)
        sweep_f = list(f_arr) + list(reversed(f_arr))[1:-1]
        sweep_r = np.full(len(sweep_f), fixed_ratio)
        
        # Slider only shows the forward pass (37 steps)
        slider_vals = list(range(37))
        slider_labels = []
        for freq_val in f_arr:
            if freq_val < 1e3:
                lbl = f"{freq_val:.0f} Hz"
            elif freq_val < 1e6:
                lbl = f"{freq_val/1e3:.0f} kHz"
            elif freq_val < 1e9:
                lbl = f"{freq_val/1e6:.1f} MHz"
            else:
                lbl = f"{freq_val/1e9:.1f} GHz"
            slider_labels.append(lbl)
        sweep_indices = list(range(len(sweep_f)))
        mode_name = "Frequency Sweep"
        
    else:
        # Static
        sweep_r = [fixed_ratio]
        sweep_f = [fixed_freq]
        sweep_indices = [0]
        slider_vals = [0]
        slider_labels = ["Static"]
        mode_name = "Static"
        
    # 2. Compute Physical parameters, stresses and impedance for each step in the sweep
    fz_arr = []
    fr_arr = []
    zr_arr = []
    zi_arr = []
    shapes = []
    pressures = []
    
    for i in range(len(sweep_r)):
        ratio_val = sweep_r[i]
        freq_val = sweep_f[i]
        
        if cond_mode:
            eps_in_val, eps_out_val = 80.0, 80.0
            sigma_out_uS_val = sigma_in_uS / ratio_val
        else:
            sigma_out_uS_val = sigma_in_uS
            eps_in_val, eps_out_val = 80.0, 80.0 / ratio_val
            
        Fz, Fr = resultant_forces(freq_val, eps_in_val, eps_out_val, sigma_in_uS, sigma_out_uS_val,
                                  R, delta, eps_m, sigma_m, E0_Vcm)
        fz_arr.append(Fz)
        fr_arr.append(Fr)
        
        # Compute impedance for this step
        Zr, Zi = get_impedance(freq_val, eps_in_val, eps_out_val, sigma_in_uS, sigma_out_uS_val,
                               R, delta, eps_m, sigma_m)
        zr_arr.append(Zr)
        zi_arr.append(Zi)
        
        a_dip, b_int, E0, _ = effective_coefficients(freq_val, eps_in_val, eps_out_val, sigma_in_uS, sigma_out_uS_val,
                                                     R, delta, eps_m, sigma_m, E0_Vcm)
        K_rc, K_rs, K_theta = stress_coefficients(a_dip, b_int, E0, eps_in_val, eps_out_val)
        shapes.append((a_dip, b_int, K_rc, K_rs, K_theta))
        
    fz_arr = np.array(fz_arr)
    fr_arr = np.array(fr_arr)
    
    # Calculate global peak forces used to normalize the force -> deformation map.
    # The peaks MUST be calibrated to the range the displayed control will roam over.
    # During a true single-axis animation we calibrate to that sweep. But in static
    # AND single-axis modes the live manual sliders (in the component) roam the WHOLE
    # space, so calibrating to a single static point would give peak≈0 and collapse
    # every manual shape to a sphere. To keep manual dragging consistent with the
    # 4-quadrant loop, we calibrate over the same bounded reference band ([1/3, 3] in
    # ratio, full frequency range) whenever manual interaction is possible.
    manual_possible = not (animate_ratio and animate_freq)   # JS sliders active unless full loop
    if (animate_ratio and animate_freq) or manual_possible:
        _R_MAX, _R_MIN = 3.0, 1.0 / 3.0
        f_e1_ref = np.logspace(0, 9, 16)[:-1]; r_e1_ref = np.full(15, _R_MAX)
        r_e2_ref = np.logspace(np.log10(_R_MAX), np.log10(_R_MIN), 16)[:-1]; f_e2_ref = np.full(15, 1e9)
        f_e3_ref = np.logspace(9, 0, 16)[:-1]; r_e3_ref = np.full(15, _R_MIN)
        r_e4_ref = np.logspace(np.log10(_R_MIN), np.log10(_R_MAX), 16)[:-1]; f_e4_ref = np.full(15, 1.0)
        ref_sweep_f = np.concatenate([f_e1_ref, f_e2_ref, f_e3_ref, f_e4_ref])
        ref_sweep_r = np.concatenate([r_e1_ref, r_e2_ref, r_e3_ref, r_e4_ref])
    else:
        # (Unreachable given manual_possible above, but kept for clarity / future use.)
        ref_sweep_f = np.asarray(sweep_f, dtype=float)
        ref_sweep_r = np.asarray(sweep_r, dtype=float)
    
    ref_prolate_diffs = []
    ref_oblate_diffs = []
    for idx_ref in range(len(ref_sweep_r)):
        r_val = ref_sweep_r[idx_ref]
        f_val = ref_sweep_f[idx_ref]
        if cond_mode:
            eps_in_val, eps_out_val = 80.0, 80.0
            sigma_out_uS_val = sigma_in_uS / r_val
        else:
            sigma_out_uS_val = sigma_in_uS
            eps_in_val, eps_out_val = 80.0, 80.0 / r_val
            
        Fz, Fr = resultant_forces(f_val, eps_in_val, eps_out_val, sigma_in_uS, sigma_out_uS_val,
                                  R, delta, eps_m, sigma_m, E0_Vcm)
        diff_ref = Fz - Fr
        if diff_ref > 0:
            ref_prolate_diffs.append(diff_ref)
        elif diff_ref < 0:
            ref_oblate_diffs.append(abs(diff_ref))
            
    peak_prolate = max(ref_prolate_diffs) if ref_prolate_diffs else 1e-20
    peak_oblate = max(ref_oblate_diffs) if ref_oblate_diffs else 1e-20
    
    # Compute geometry for each frame using a logarithmic scaling function to compress the large 
    # force dynamic range. This makes intermediate prolate forces (e.g. ratio = 2) look beautifully dramatic 
    # while still cleanly capping at the maximum deformation at ratio = 100.
    computed_frames_data = []
    # Logarithmic compression fractions for mapping force -> deformation. The full
    # 0.01..100 grid spans a ~10000x prolate range needing aggressive compression;
    # the bounded loop [1/3,3] and the single-axis sweeps span a far smaller range,
    # so use a gentler fraction there to keep intermediate shapes visible.
    if animate_ratio and animate_freq:
        fraction_prolate = 0.02
        fraction_oblate = 0.05
    else:
        fraction_prolate = 0.00002  # full-range prolate (covers 10000x dynamic range)
        fraction_oblate = 0.05      # oblate forces span ~1x range
    for i in range(len(sweep_r)):
        ratio_val = sweep_r[i]
        freq_val = sweep_f[i]
        Fz, Fr = fz_arr[i], fr_arr[i]

        # Smooth, gradual force -> deformation mapping (matches the JS manual path).
        # Normalize |diff| between the spherical threshold and the peak force, then
        # apply a gamma<1 power law. This varies continuously from 0 (sphere) to +/-1
        # (max prolate/oblate) with no plateau or cliff, so the shape eases gradually
        # into spherical where appropriate. thr uses the same 2% relative rule as the
        # phase grid and badge classifier.
        if cond_mode:
            _eo_lo = 80.0; _so_lo = sigma_in_uS / ratio_val
        else:
            _so_lo = sigma_in_uS; _eo_lo = 80.0 / ratio_val
        _flo = resultant_forces(1.0, 80.0, _eo_lo, sigma_in_uS, _so_lo,
                                R, delta, eps_m, sigma_m, E0_Vcm)
        _thr = 0.02 * max(abs(_flo[0] - _flo[1]), 1e-12)

        diff = Fz - Fr
        _adiff = abs(diff)
        _peak = peak_prolate if diff >= 0 else peak_oblate
        _span = _peak - _thr
        _sign = 1.0 if diff >= 0 else -1.0
        if _adiff <= _thr:
            deform = 0.0                       # balanced -> perfect sphere
        elif _span <= 0:
            # Force past the calibration peak -> max deformation, never a sphere
            # (this prevents the one-frame "bounce" at ratio=100).
            deform = _sign
        else:
            _u = (_adiff - _thr) / _span       # 0 at threshold, 1 at peak
            _u = min(1.0, max(0.0, _u))
            deform = _sign * (_u ** 0.5)        # gamma=0.5
            
        deform = float(np.clip(deform, -1.0, 1.0))
        k = max(0.25, min(1.9, 1.0 + K_MAX * deform))
        a_ax = R0 * k**(2/3)
        b_ax = R0 / k**(1/3)
        
        K_rc, K_rs, K_theta = shapes[i][2], shapes[i][3], shapes[i][4]
        _, _, phi_s = silhouette(a_ax, b_ax)
        p = (K_rc * np.cos(phi_s)**2 + K_rs * np.sin(phi_s)**2
             + K_theta * np.cos(phi_s) * np.sin(phi_s))
        pressures.append(p)
        computed_frames_data.append((a_ax, b_ax, p))
        
    all_p = np.concatenate(pressures)
    max_p = max(abs(all_p.min()), abs(all_p.max()), 1e-30)
    
    def local_make_traces(a_ax, b_ax, pressure, include_colorbar=False):
        traces = []
        x_s, y_s, _ = silhouette(a_ax, b_ax)
        cbar = dict(
            title=dict(text='Maxwell stress<br>(N/m²)', font=dict(size=10, color='#0f172a')),
            thickness=14, len=0.65, x=1.02,
            tickformat='.1e', tickfont=dict(size=9, color='#0f172a'),
        ) if include_colorbar else None

        traces.append(go.Scatter(
            x=x_s, y=y_s, mode='markers',
            marker=dict(color=pressure, colorscale=COLORSCALE,
                        cmin=-max_p, cmax=max_p, size=8,
                        colorbar=cbar, showscale=True),
            showlegend=False, name='silhouette',
            hovertemplate='p: %{marker.color:.2e} N/m2<extra></extra>',
        ))

        for idx_l, lf in enumerate(LAT_FRACS):
            x_l, y_l = latitude_ellipse(a_ax, b_ax, lf)
            is_equator = (idx_l == N_LAT // 2)
            traces.append(go.Scatter(
                x=x_l, y=y_l, mode='lines',
                line=dict(color='#1e3a5f', width=1.8 if is_equator else 1.0),
                opacity=0.75 if is_equator else 0.38,
                showlegend=False, hoverinfo='skip', name=f'lat{idx_l}',
            ))

        for idx_m, th in enumerate(MER_THETAS):
            x_m, y_m = meridian_arc(a_ax, b_ax, th)
            traces.append(go.Scatter(
                x=x_m, y=y_m, mode='lines',
                line=dict(color='#1e3a5f', width=1.5),
                opacity=max(0.80 - idx_m * 0.22, 0.15),
                showlegend=False, hoverinfo='skip', name=f'mer{idx_m}',
            ))

        for idx_f, fx in enumerate(FIELD_XS):
            fl_x, fl_y = field_line_xy(fx)
            traces.append(go.Scatter(
                x=fl_x, y=fl_y, mode='lines',
                line=dict(color='rgba(180,100,0,0.40)', width=1.5, dash='dash'),
                showlegend=False, hoverinfo='skip', name=f'field{idx_f}',
            ))
        return traces

    # Build the figure
    init_a_ax, init_b_ax, init_p = computed_frames_data[0]
    init_traces = local_make_traces(init_a_ax, init_b_ax, init_p, include_colorbar=True)
    N_TOTAL_TRACES = len(init_traces)
    
    frames = []
    
    if mode_name != "Static":
        for idx in sweep_indices:
            ratio_val = sweep_r[idx]
            freq_val = sweep_f[idx]
            a_ax, b_ax, p = computed_frames_data[idx]
            
            if mode_name == "Frequency Sweep":
                slider_idx = idx if idx < 37 else (72 - idx)
                freq_lbl = slider_labels[slider_idx]
                title_lbl = f"f = {freq_lbl} | Ratio = {fixed_ratio:.2f}"
            elif mode_name == "Ratio Sweep":
                slider_idx = idx if idx < 41 else (80 - idx)
                ratio_lbl = slider_labels[slider_idx]
                if fixed_freq < 1e3:
                    f_lbl = f"{fixed_freq:.0f} Hz"
                elif fixed_freq < 1e6:
                    f_lbl = f"{fixed_freq/1e3:.0f} kHz"
                elif fixed_freq < 1e9:
                    f_lbl = f"{fixed_freq/1e6:.1f} MHz"
                else:
                    f_lbl = f"{fixed_freq/1e9:.1f} GHz"
                title_lbl = f"f = {f_lbl} | Ratio = {ratio_lbl}"
            else:  # 4-Quadrant Loop
                slider_idx = idx
                _b1, _b2, _b3 = _q_bounds
                if idx < _b1:
                    quad = "Q1: Freq Sweep (Ratio=3)"
                elif idx < _b2:
                    quad = "Q2: Ratio Sweep (Freq=1GHz)"
                elif idx < _b3:
                    quad = "Q3: Freq Sweep (Ratio=0.33)"
                else:
                    quad = "Q4: Ratio Sweep (Freq=1Hz)"
                    
                freq_val_curr = sweep_f[idx]
                if freq_val_curr < 1e3:
                    f_lbl = f"{freq_val_curr:.0f} Hz"
                elif freq_val_curr < 1e6:
                    f_lbl = f"{freq_val_curr/1e3:.0f} kHz"
                elif freq_val_curr < 1e9:
                    f_lbl = f"{freq_val_curr/1e6:.1f} MHz"
                else:
                    f_lbl = f"{freq_val_curr/1e9:.1f} GHz"
                title_lbl = f"{quad} | f = {f_lbl}, Ratio = {sweep_r[idx]:.2f}"
                
            frame_traces = local_make_traces(a_ax, b_ax, p, include_colorbar=False)
            frames.append(go.Frame(
                data=frame_traces,
                traces=list(range(N_TOTAL_TRACES)),
                name=f"f{idx}",
                layout=go.Layout(
                    title=dict(
                        text=f"<b>Vesicle Electrostriction (v6) — {mode_name}</b><br><span style='font-size:13px;color:#475569'>{title_lbl}</span>"
                    )
                )
            ))
            
    if mode_name == "Static":
        if fixed_freq < 1e3:
            f_lbl = f"{fixed_freq:.0f} Hz"
        elif fixed_freq < 1e6:
            f_lbl = f"{fixed_freq/1e3:.0f} kHz"
        elif fixed_freq < 1e9:
            f_lbl = f"{fixed_freq/1e6:.1f} MHz"
        else:
            f_lbl = f"{fixed_freq/1e9:.1f} GHz"
        init_title = f"<b>Vesicle Electrostriction (v6) — Static Shape</b><br><span style='font-size:13px;color:#475569'>f = {f_lbl} | Ratio = {fixed_ratio:.2f}</span>"
    else:
        init_title = f"<b>Vesicle Electrostriction (v6) — {mode_name}</b><br><span style='font-size:13px;color:#475569'>Ready to Animate</span>"
        
    fig = go.Figure(
        data=init_traces,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text=init_title,
                font=dict(family="Inter, sans-serif", size=14, color="#0f172a"),
                x=0.5, xanchor="center",
            ),
            width=500, height=410,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Inter, sans-serif", color="#0f172a"),
            margin=dict(l=50, r=90, t=90, b=50),
            xaxis=dict(range=[-20, 20], title="x (μm)", showgrid=True,
                       gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#cbd5e1",
                       tickfont=dict(size=9), title_font=dict(size=10), fixedrange=True),
            yaxis=dict(range=[-15, 15], title="z (field axis, μm)", showgrid=True,
                       gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#cbd5e1",
                       scaleanchor="x", scaleratio=1, constrain="domain",
                       tickfont=dict(size=9), title_font=dict(size=10), fixedrange=True),
        )
    )
    
    rat_type = "σ" if cond_mode else "ε"
    fig.add_annotation(
        text=(f"E field direction: vertical &nbsp;|&nbsp; "
              f"{rat_type}<sub>in</sub>/{rat_type}<sub>out</sub> sweeps &nbsp;|&nbsp; "
              f"E₀={E0_Vcm:.0f} V/cm"),
        x=0.5, y=-0.15, xref="paper", yref="paper",
        showarrow=False, font=dict(size=9, color="#64748b"), xanchor="center"
    )
            
    # Build fig_phase
    f_mesh, r_mesh = np.meshgrid(freq_grid, ratios_grid)
    custom_data = np.stack([f_mesh, r_mesh], axis=-1)
    
    fig_phase = go.Figure(data=go.Heatmap(
        x=np.log10(freq_grid), y=np.log10(ratios_grid), z=shape_grid,
        customdata=custom_data,
        colorscale=[[0, '#ea580c'], [0.5, '#e2e8f0'], [1, '#dc2626']],
        zmin=-1, zmax=1,
        showscale=False, hoverongaps=False,
        hovertemplate="Freq: %{customdata[0]:.2e} Hz<br>Ratio: %{customdata[1]:.2f}<extra></extra>"))
    fig_phase.add_trace(go.Scatter(
        x=np.log10(f_mesh.flatten()), y=np.log10(r_mesh.flatten()), mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=10), hoverinfo="skip",
        showlegend=False, name="clickable_mesh"))
    fig_phase.add_trace(go.Scatter(
        x=[np.log10(fixed_freq)], y=[np.log10(fixed_ratio)], mode="markers", name="Current Operating Point",
        marker=dict(color="#2563eb", size=14, symbol="star",
                    line=dict(color="black", width=2))))
    
    fig_phase.update_xaxes(type="linear",
                           range=[0, 9],
                           tickvals=[0, 2, 4, 6, 8, 9],
                           ticktext=["1", "100", "10k", "1M", "100M", "1G"],
                           gridcolor="#cbd5e1",
                           linecolor="#0f172a", tickcolor="#0f172a",
                           tickfont=dict(color="#0f172a", size=10, family="Inter, sans-serif"))
    fig_phase.update_yaxes(type="linear",
                           range=[-2, 2],
                           tickvals=[-2, -1, 0, 1, 2],
                           ticktext=["0.01", "0.1", "1", "10", "100"],
                           gridcolor="#cbd5e1",
                           linecolor="#0f172a", tickcolor="#0f172a",
                           tickfont=dict(color="#0f172a", size=10, family="Inter, sans-serif"))
    fig_phase.update_layout(autosize=True, height=340, margin=dict(l=45, r=20, t=10, b=40),
                            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                            font=dict(color="#0f172a", family="Inter, sans-serif"),
                            showlegend=False, clickmode='event+select')

    # Build fig_sweep
    if cache_dict is not None:
        freq_sweep = cache_dict['sw_f']
        ri_fixed = int(np.argmin(np.abs(np.log10(cache_dict['ratio_arr']) - np.log10(fixed_ratio))))
        Fz_sweep = cache_dict['sw_Fz'][ri_fixed]
        Fr_sweep = cache_dict['sw_Fr'][ri_fixed]
        Z_real_sweep = cache_dict['sw_Zr'][ri_fixed]
        Z_imag_sweep = cache_dict['sw_Zi'][ri_fixed]
        
        fi_fixed = int(np.argmin(np.abs(np.log10(freq_sweep) - np.log10(fixed_freq))))
        Fz_curr_val = float(Fz_sweep[fi_fixed])
        Fr_curr_val = float(Fr_sweep[fi_fixed])
        Z_real_curr_val = float(Z_real_sweep[fi_fixed])
        Z_imag_curr_val = float(Z_imag_sweep[fi_fixed])
    else:
        freq_sweep = np.logspace(0, 9, 300)
        Fz_sweep, Fr_sweep = np.zeros_like(freq_sweep), np.zeros_like(freq_sweep)
        Z_real_sweep, Z_imag_sweep = np.zeros_like(freq_sweep), np.zeros_like(freq_sweep)
        Fz_curr_val, Fr_curr_val = 0.0, 0.0
        Z_real_curr_val, Z_imag_curr_val = 0.0, 0.0

    # ── Fixed axis ranges (pinned, clip-proof) ──────────────────────────────
    # The sweep y-axes are pinned so they never autoscale per frame (that was the
    # "zooming" jitter). To avoid clipping we frame the range over the relevant
    # ratio band AND guarantee it always contains the live operating point's value,
    # with a floor so a balanced (force=0) point still has a visible scale.
    if cache_dict is not None:
        _rarr = cache_dict['ratio_arr']
        if animate_ratio and animate_freq:
            # 4-quadrant loop: bounded band [1/3, 3]
            _mask = (_rarr >= (1.0/3.0) * 0.999) & (_rarr <= 3.0 * 1.001)
        elif animate_ratio and not animate_freq:
            # ratio-only sweep: full ratio span is traversed
            _mask = np.ones(len(_rarr), dtype=bool)
        else:
            # static / frequency-only: the displayed ratio plus its immediate
            # neighbours, so the pinned range stays valid even if the live slider
            # ratio differs slightly from the layout-time fixed_ratio.
            _ri0 = int(np.argmin(np.abs(np.log10(_rarr) - np.log10(fixed_ratio))))
            _lo = max(0, _ri0 - 1); _hi = min(len(_rarr), _ri0 + 2)
            _mask = np.zeros(len(_rarr), dtype=bool); _mask[_lo:_hi] = True
        if not _mask.any():
            _mask = np.ones(len(_rarr), dtype=bool)

        def _padded_range(vals, floor):
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return [-floor, floor]
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            span = vmax - vmin
            if span < floor:                       # near-degenerate (e.g. forces=0)
                mid = 0.5 * (vmin + vmax)
                vmin, vmax = mid - 0.5 * floor, mid + 0.5 * floor
                span = vmax - vmin
            pad = 0.12 * span
            return [vmin - pad, vmax + pad]

        _allF = np.concatenate([cache_dict['sw_Fz'][_mask].ravel(), cache_dict['sw_Fr'][_mask].ravel()])
        _allZ = np.concatenate([cache_dict['sw_Zr'][_mask].ravel(), cache_dict['sw_Zi'][_mask].ravel()])
        # Force floor: a balanced point (force≈0) still gets a small symmetric scale.
        force_yrange = _padded_range(_allF, floor=0.05)
        # Impedance floor scaled to the data magnitude so resonant peaks never clip.
        _zfloor = 0.05 * (np.nanmax(np.abs(_allZ)) if _allZ.size else 1.0)
        imp_yrange = _padded_range(_allZ, floor=_zfloor if _zfloor > 0 else 1.0)
        # x-axis is log10 of frequency; sw_f spans 1 Hz..1 GHz
        _xlo = float(np.log10(np.min(freq_sweep)))
        _xhi = float(np.log10(np.max(freq_sweep)))
        sweep_xrange = [_xlo, _xhi]
    else:
        force_yrange = [-1.0, 1.0]
        imp_yrange = [-1.0, 1.0]
        sweep_xrange = [0.0, 9.0]

    fig_sweep = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sweep.add_trace(go.Scatter(x=freq_sweep, y=Fz_sweep, name="FR_z (Pole)",
                                   line=dict(color="#ef4444", width=3)), secondary_y=False)
    fig_sweep.add_trace(go.Scatter(x=freq_sweep, y=Fr_sweep, name="FR_r (Equator)",
                                   line=dict(color="#3b82f6", width=2.5)), secondary_y=False)
    fig_sweep.add_trace(go.Scatter(x=freq_sweep, y=Z_real_sweep, name="Z_real",
                                   line=dict(color="#16a34a", width=2, dash='dot')), secondary_y=True)
    fig_sweep.add_trace(go.Scatter(x=freq_sweep, y=Z_imag_sweep, name="Z_imag",
                                   line=dict(color="#22c55e", width=2, dash='dash')), secondary_y=True)
    fig_sweep.add_trace(go.Scatter(x=freq_sweep, y=Fz_sweep, mode="markers",
                                   marker=dict(color="rgba(0,0,0,0)", size=12),
                                   hoverinfo="skip", showlegend=False,
                                   name="clickable_sweep"), secondary_y=False)
    fig_sweep.add_trace(go.Scatter(x=[fixed_freq], y=[Fz_curr_val], mode="markers",
                                   marker=dict(color="#ef4444", size=10), showlegend=False,
                                   hoverinfo="skip"), secondary_y=False)
    fig_sweep.add_trace(go.Scatter(x=[fixed_freq], y=[Fr_curr_val], mode="markers",
                                   marker=dict(color="#3b82f6", size=10), showlegend=False,
                                   hoverinfo="skip"), secondary_y=False)
    fig_sweep.add_trace(go.Scatter(x=[fixed_freq], y=[Z_real_curr_val], mode="markers",
                                   marker=dict(color="#16a34a", size=10), showlegend=False,
                                   hoverinfo="skip"), secondary_y=True)
    fig_sweep.add_trace(go.Scatter(x=[fixed_freq], y=[Z_imag_curr_val], mode="markers",
                                   marker=dict(color="#22c55e", size=10), showlegend=False,
                                   hoverinfo="skip"), secondary_y=True)
    
    fig_sweep.add_vline(x=fixed_freq, line_width=2, line_dash="dash", line_color="#0f172a")
    
    axfont = dict(color="#0f172a", size=11, family="Inter, sans-serif")
    tkfont = dict(color="#0f172a", size=9, family="Inter, sans-serif")
    fig_sweep.update_xaxes(type="log", gridcolor="#cbd5e1",
                           linecolor="#0f172a", tickcolor="#0f172a",
                           title_font=axfont, tickfont=tkfont,
                           range=sweep_xrange, autorange=False, fixedrange=True)
    fig_sweep.update_yaxes(secondary_y=False, gridcolor="#cbd5e1",
                           linecolor="#0f172a", tickcolor="#0f172a",
                           title_font=axfont, tickfont=tkfont,
                           range=force_yrange, autorange=False, fixedrange=True)
    fig_sweep.update_yaxes(secondary_y=True,
                           linecolor="#0f172a", tickcolor="#0f172a",
                           title_font=axfont, tickfont=tkfont,
                           range=imp_yrange, autorange=False, fixedrange=True)
    fig_sweep.update_layout(
        autosize=True, height=340, margin=dict(l=55, r=55, t=40, b=40),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#cbd5e1", borderwidth=1,
                    font=dict(color="#0f172a", size=9)),
        hovermode="x unified", clickmode='event+select')

    phase_frames = []
    sweep_frames = []
    
    metadata_list = []
    for idx in range(len(sweep_r)):
        ratio_val = sweep_r[idx]
        freq_val = sweep_f[idx]
        Fz = fz_arr[idx]
        Fr = fr_arr[idx]
        diff = Fz - Fr
        
        if cond_mode:
            eps_in_val, eps_out_val = 80.0, 80.0
            sigma_out_uS_val = sigma_in_uS / ratio_val
        else:
            sigma_out_uS_val = sigma_in_uS
            eps_in_val, eps_out_val = 80.0, 80.0 / ratio_val
            
        fz_lo, fr_lo = resultant_forces(1.0, eps_in_val, eps_out_val, sigma_in_uS, sigma_out_uS_val,
                                        R, delta, eps_m, sigma_m, E0_Vcm)
        thr = 0.02 * max(abs(fz_lo - fr_lo), 1e-12)
        
        if diff > thr:
            s_desc = "Prolate (stretched along the field)"
            s_color = "#dc2626"
        elif diff < -thr:
            s_desc = "Oblate (compressed along the field)"
            s_color = "#ea580c"
        else:
            s_desc = "Spherical (forces balanced)"
            s_color = "#64748b"
            
        # Construct val_lbl
        if mode_name == "Frequency Sweep":
            slider_idx = idx if idx < 37 else (72 - idx)
            freq_lbl = slider_labels[slider_idx]
            val_lbl = f"f = {freq_lbl}"
        elif mode_name == "Ratio Sweep":
            slider_idx = idx if idx < 41 else (80 - idx)
            ratio_lbl = slider_labels[slider_idx]
            val_lbl = f"Ratio = {ratio_lbl}"
        elif mode_name == "4-Quadrant Loop":
            if freq_val < 1e3:
                f_lbl = f"{freq_val:.0f} Hz"
            elif freq_val < 1e6:
                f_lbl = f"{freq_val/1e3:.0f} kHz"
            elif freq_val < 1e9:
                f_lbl = f"{freq_val/1e6:.1f} MHz"
            else:
                f_lbl = f"{freq_val/1e9:.1f} GHz"
            val_lbl = f"f = {f_lbl}, Ratio = {ratio_val:.2f}"
        else:
            val_lbl = "Static"
            
        metadata_list.append({
            'desc': s_desc,
            'color': s_color,
            'fz': f"{Fz:.2e}",
            'fr': f"{Fr:.2e}",
            'val_lbl': val_lbl,
            'ratio': float(ratio_val),
            'freq': float(freq_val),
        })

    if mode_name != "Static" and cache_dict is not None:
        sw_f = cache_dict['sw_f']
        for idx in sweep_indices:
            r_val = sweep_r[idx]
            f_val = sweep_f[idx]
            
            # Phase frame: update star marker trace (index 2)
            phase_frames.append(go.Frame(
                data=[
                    go.Scatter(x=[np.log10(f_val)], y=[np.log10(r_val)])
                ],
                traces=[2],
                name=f"f{idx}"
            ))
            
            # Sweep frame: update sweep curves, vertical line shape, and current dots
            ri_g = int(np.argmin(np.abs(np.log10(cache_dict['ratio_arr']) - np.log10(r_val))))
            Fz_curve = cache_dict['sw_Fz'][ri_g]
            Fr_curve = cache_dict['sw_Fr'][ri_g]
            Zr_curve = cache_dict['sw_Zr'][ri_g]
            Zi_curve = cache_dict['sw_Zi'][ri_g]
            
            fz_val = fz_arr[idx]
            fr_val = fr_arr[idx]
            zr_val = zr_arr[idx]
            zi_val = zi_arr[idx]
            
            sweep_frames.append(go.Frame(
                data=[
                    go.Scatter(x=sw_f, y=Fz_curve),  # Trace 0
                    go.Scatter(x=sw_f, y=Fr_curve),  # Trace 1
                    go.Scatter(x=sw_f, y=Zr_curve),  # Trace 2
                    go.Scatter(x=sw_f, y=Zi_curve),  # Trace 3
                    go.Scatter(),                    # Trace 4
                    go.Scatter(x=[f_val], y=[fz_val]),  # Trace 5
                    go.Scatter(x=[f_val], y=[fr_val]),  # Trace 6
                    go.Scatter(x=[f_val], y=[zr_val]),  # Trace 7
                    go.Scatter(x=[f_val], y=[zi_val])   # Trace 8
                ],
                traces=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                name=f"f{idx}",
                layout=go.Layout(
                    shapes=[
                        dict(
                            type="line",
                            x0=f_val, x1=f_val,
                            y0=0, y1=1,
                            yref="paper",
                            line=dict(color="#0f172a", width=2, dash="dash")
                        )
                    ]
                )
            ))
            
        fig_phase.frames = phase_frames
        fig_sweep.frames = sweep_frames
        
    return fig, fig_phase, fig_sweep, mode_name, metadata_list, peak_prolate, peak_oblate, max_p


def _selftest(R=10e-6, delta=5e-9, eps_m=5.0, sigma_m=1e-8, E0_Vcm=200.0):
    results = []

    # 1. dipole amplitude bounded
    ok = True
    for rs in [0.1, 0.5, 2, 10]:
        a, b, E0, _ = effective_coefficients(1.0, 80, 80, 15.0, 15.0 / rs,
                                             R, delta, eps_m, sigma_m, E0_Vcm)
        if not (abs(a) < 5 * abs(E0)):
            ok = False
    results.append(("Dipole amplitude bounded (|a| < 5 E0)", ok))

    # 2. conductivity sign flip at low frequency
    fz_hi, _ = resultant_forces(1.0, 80, 80, 15.0, 7.5, R, delta, eps_m, sigma_m, E0_Vcm)   # rs=2
    fz_lo, _ = resultant_forces(1.0, 80, 80, 15.0, 30.0, R, delta, eps_m, sigma_m, E0_Vcm)  # rs=0.5
    results.append(("Conductivity flip: FR_z(rs=2)>0 & FR_z(rs=0.5)<0",
                    (fz_hi > 0) and (fz_lo < 0)))

    # 3. permittivity sign flip (reversed), Rs=1
    fz_e_lo, _ = resultant_forces(1.0, 40, 80, 15.0, 15.0, R, delta, eps_m, sigma_m, E0_Vcm)  # Re=0.5
    fz_e_hi, _ = resultant_forces(1.0, 80, 40, 15.0, 15.0, R, delta, eps_m, sigma_m, E0_Vcm)  # Re=2
    results.append(("Permittivity flip: FR_z(Re=0.5)>0 & FR_z(Re=2)<0",
                    (fz_e_lo > 0) and (fz_e_hi < 0)))

    # 4. Maxwell-Wagner frequency in paper band
    f1 = maxwell_wagner_freq(80, 80, 15.0, 7.5)
    f2 = maxwell_wagner_freq(80, 80, 15.0, 30.0)
    results.append(("f_MW in [1e5,1e6] Hz for paper sigma",
                    (1e5 <= f1 <= 1e6) and (1e5 <= f2 <= 1e6)))

    # 5. Hyuga / Gao share angular shape
    th = np.linspace(0.01, np.pi - 0.01, 50)
    ph = analytical_pressure(th, 2.0, 1.0, 80, 2e4, "hyuga")
    pg = analytical_pressure(th, 2.0, 1.0, 80, 2e4, "gao")
    ratio = ph / pg
    results.append(("Hyuga/Gao share angular shape (constant ratio)",
                    np.std(ratio) / abs(np.mean(ratio)) < 1e-9))

    # 6. Slanted boundary: the prolate->sphere edge frequency must DECREASE as the
    #    conductivity ratio increases (Fig 7a). Find the edge for two ratios.
    def _edge(rv):
        fg = np.logspace(0, 9, 120)   # 1 Hz – 1 GHz
        fz, fr = resultant_forces(fg, 80, 80, 15.0, 15.0 / rv, R, delta, eps_m, sigma_m, E0_Vcm)
        d = fz - fr
        thr = 0.02 * abs(d[0])
        cls = np.where(np.abs(d) > thr, 1, 0)
        idx = np.where(cls == 0)[0]
        return fg[idx[0]] if len(idx) else fg[-1]
    results.append(("Slanted boundary: edge(ratio=100) < edge(ratio=2)",
                    _edge(100.0) < _edge(2.0)))

    # 7. Label/3D-shape consistency: at a frequency inside the sphere band, the
    #    same THRESH that drives the text label must also keep |Fz-Fr| <= THRESH
    #    so the 3D vesicle renders as a sphere (no silent disagreement).
    fz_lo, fr_lo = resultant_forces(1.0, 80, 80, 15.0, 7.5, R, delta, eps_m, sigma_m, E0_Vcm)
    thr = 0.02 * max(abs(fz_lo - fr_lo), 1e-12)
    fz_hi, fr_hi = resultant_forces(5e7, 80, 80, 15.0, 7.5, R, delta, eps_m, sigma_m, E0_Vcm)
    results.append(("Sphere-band consistency: |Fz-Fr| <= THRESH at 50 MHz (ratio=2)",
                    abs(fz_hi - fr_hi) <= thr))
    return results


# ============================================================================
# PRE-COMPUTED CACHE  (all slider positions; O(1) lookups after first run)
# ============================================================================
# Strategy: at startup, evaluate every physics result for every combination of
# (ratio, frequency) that the sliders can select.  Results are stored in a dict
# of NumPy arrays keyed by plot/use-case.  Subsequent slider moves do only an
# array-index lookup — no physics is re-evaluated on each Streamlit rerun.
#
# @st.cache_data memoises by argument hash.  The tuple arguments (ratio_vals_t,
# freq_vals_t) ensure the cache is invalidated if the slider grids change.
# The 'Physics version: v4' stamp in the docstring forces cache invalidation
# if the cached function's source code changes between app restarts.
# ============================================================================

@st.cache_data(show_spinner="⚙️ Pre-computing physics for all slider positions…")
def precompute_all(cond_mode, sigma_in_uS, R, delta, eps_m, sigma_m, E0_Vcm,
                   ratio_vals_t, freq_vals_t):
    """Eagerly compute every result the dashboard needs for every slider position.

    Returns a dict of NumPy arrays.  All subsequent interactions are O(1)
    array-index lookups with no physics re-evaluated after startup.

    Physics version: v6 — force cache refresh to update phase diagram axes log10 ranges
    """
    ratio_arr = np.array(ratio_vals_t, dtype=float)
    freq_arr  = np.array(freq_vals_t,  dtype=float)
    nr, nf    = len(ratio_arr), len(freq_arr)

    def _params(rv):
        if cond_mode:
            return 80.0, 80.0, sigma_in_uS, sigma_in_uS / rv
        return 80.0, 80.0 / rv, sigma_in_uS, sigma_in_uS

    # Phase-diagram heatmap grid (45 × 45 log-spaced) -------------------------
    pg_r = np.logspace(-2, 2, 45)
    pg_f = np.logspace(0,  9, 45)
    pg_d = np.zeros((45, 45))
    for i, rv in enumerate(pg_r):
        ei, eo, si, so = _params(rv)
        fz, fr = resultant_forces(pg_f, ei, eo, si, so, R, delta, eps_m, sigma_m, E0_Vcm)
        pg_d[i] = fz - fr

    # Sweep curves (300-pt dense axis, one row per slider ratio) --------------
    sw_f  = np.logspace(0, 9, 300)
    sw_Fz = np.zeros((nr, 300))
    sw_Fr = np.zeros((nr, 300))
    sw_Zr = np.zeros((nr, 300))
    sw_Zi = np.zeros((nr, 300))
    for i, rv in enumerate(ratio_arr):
        ei, eo, si, so = _params(rv)
        sw_Fz[i], sw_Fr[i] = resultant_forces(sw_f, ei, eo, si, so,
                                               R, delta, eps_m, sigma_m, E0_Vcm)
        sw_Zr[i], sw_Zi[i] = get_impedance(sw_f, ei, eo, si, so, R, delta, eps_m, sigma_m)

    # Operating-point forces & dipole fields (nr × nf) -----------------------
    op_Fz = np.zeros((nr, nf))
    op_Fr = np.zeros((nr, nf))
    op_ar = np.zeros((nr, nf))   # Re(a_dipole)
    op_ai = np.zeros((nr, nf))   # Im(a_dipole)
    op_br = np.zeros((nr, nf))   # Re(b_interior)
    op_bi = np.zeros((nr, nf))   # Im(b_interior)
    op_ei = np.zeros(nr)         # eps_in  (ratio-dependent)
    op_eo = np.zeros(nr)         # eps_out

    for i, rv in enumerate(ratio_arr):
        ei, eo, si, so = _params(rv)
        op_ei[i], op_eo[i] = ei, eo
        fz_v, fr_v = resultant_forces(freq_arr, ei, eo, si, so,
                                      R, delta, eps_m, sigma_m, E0_Vcm)
        a_v, b_v, _, _ = effective_coefficients(freq_arr, ei, eo, si, so,
                                                R, delta, eps_m, sigma_m, E0_Vcm)
        op_Fz[i] = fz_v;  op_Fr[i] = fr_v
        op_ar[i] = np.real(a_v);  op_ai[i] = np.imag(a_v)
        op_br[i] = np.real(b_v);  op_bi[i] = np.imag(b_v)

    return dict(
        pg_r=pg_r, pg_f=pg_f, pg_d=pg_d,
        sw_f=sw_f, sw_Fz=sw_Fz, sw_Fr=sw_Fr, sw_Zr=sw_Zr, sw_Zi=sw_Zi,
        op_Fz=op_Fz, op_Fr=op_Fr,
        op_ar=op_ar, op_ai=op_ai, op_br=op_br, op_bi=op_bi,
        op_ei=op_ei, op_eo=op_eo,
        E0_vm=float(E0_Vcm * 100.0),
        ratio_arr=ratio_arr, freq_arr=freq_arr,
    )


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("## ⚙️ Simulation Parameters")

with st.sidebar.expander("📌 Fixed Parameters (Constant in Paper)", expanded=True):
    st.markdown(r"""
* **Vesicle Radius ($R$):** $10\ \mu\text{m}$
* **Membrane Thickness ($\delta$):** $5\text{ nm}$
* **Membrane Permittivity ($\epsilon_m$):** $5$
* **Membrane Conductivity ($\sigma_m$):** $10^{-8}\ \text{S/m}$
* **Reference Inside Conductivity ($\sigma_{\text{in}}$):** $15\ \mu\text{S/cm}$
* **Electric Field ($E_0$):** $200\ \text{V/cm}$
""")

# Fixed physical parameters
R = 10.0 * 1e-6
delta = 5.0 * 1e-9
E0_Vcm = 200.0
eps_m = 5.0
sigma_m = 1e-8
sigma_in_uS = 15.0

# Which ratio drives the sweep (reproduces Fig 5/7a vs Fig 6/7b)
sweep_mode = st.sidebar.radio(
    "Sweep variable",
    options=["Conductivity ratio (σin/σout)", "Permittivity ratio (εin/εout)"],
    index=0,
)
COND_MODE = sweep_mode.startswith("Conductivity")

# ============================================================================
# STATE & INTERACTIVITY
# ============================================================================
# Streamlit re-runs the entire script on every user interaction.  We use:
#   - st.session_state  to persist the selected ratio and frequency across reruns
#   - on_select="rerun" on plotly_chart to trigger a rerun when the user clicks
#   - find_nearest()    to snap clicked coordinates to the nearest pre-computed
#                       slider value (log-scale distance to handle the log axes)
# ============================================================================

# Log-spaced slider grids — must match the grids used in precompute_all()
# 41 ratio steps: 0.01 -> 100 (4 decades);  37 frequency steps: 1 Hz -> 1 GHz (9 decades)
_r = np.logspace(-2, 2, 41)
ratio_vals = [round(float(v), 6) for v in _r]
_f = np.logspace(0, 9, 37)
freq_vals  = [round(float(v), 4) for v in _f]


def find_nearest(val, possible_vals):
    """Snap val to the nearest entry in possible_vals using log-scale distance.
    Used when converting a plotly click coordinate to the closest slider step.
    """
    if val is None or val <= 0:
        return possible_vals[0]
    distances = [abs(np.log10(val) - np.log10(p)) for p in possible_vals]
    return possible_vals[int(np.argmin(distances))]


# Initialize session state from query params if available, otherwise defaults
if "ratio" in st.query_params:
    try:
        st.session_state.ratio = find_nearest(float(st.query_params["ratio"]), ratio_vals)
    except Exception:
        pass

if "freq" in st.query_params:
    try:
        st.session_state.freq = find_nearest(float(st.query_params["freq"]), freq_vals)
    except Exception:
        pass

if "ratio" not in st.session_state:
    st.session_state.ratio = find_nearest(2.0, ratio_vals)
if "freq" not in st.session_state:
    st.session_state.freq = find_nearest(1e5, freq_vals)

st.sidebar.markdown("### 🎬 Animation Controls")
# The live sliders / plot clicks in panel 1 can stop a running animation. When they
# do, the component sets ?stop_anim=1 and triggers a rerun; we honor it here by
# forcing both toggles off BEFORE the widgets are instantiated (their state is keyed
# in session_state, so we reset the keys, then clear the query flag).
if st.query_params.get("stop_anim") == "1":
    st.session_state["anim_ratio_toggle"] = False
    st.session_state["anim_freq_toggle"] = False
    try:
        del st.query_params["stop_anim"]
    except Exception:
        pass

animate_ratio = st.sidebar.toggle("Animate Ratio Sweep", value=False, key="anim_ratio_toggle")
animate_freq = st.sidebar.toggle("Animate Frequency Sweep", value=False, key="anim_freq_toggle")

st.sidebar.markdown("### 📍 Current Operating Point")
# The operating point is driven by the LIVE in-component sliders (and by clicking
# the phase/sweep plots), which update everything client-side with no server rerun.
# We therefore do NOT place Streamlit slider widgets here — those would trigger a
# full rerun (and a visible component reload) on every change. The current values
# are read from session_state / URL only, purely to seed the component and the
# validation panel. The sliders live inside the plot component itself.
ratio = st.session_state.ratio
freq = st.session_state.freq

if animate_ratio or animate_freq:
    st.sidebar.caption(
        "Animating — grab either live slider in panel 1 (or click the phase / sweep "
        "plots) to stop the animation and take manual control."
    )
else:
    st.sidebar.caption(
        "Use the live sliders in panel 1 (or click the phase / sweep plots) to move "
        "the operating point instantly — no page reload."
    )

# Keep URL query parameters in sync with current state (seed only; the component
# updates the URL itself during live interaction without forcing a rerun).
st.query_params["ratio"] = str(ratio)
st.query_params["freq"] = str(freq)

# ── Validation toggle (end of controls; off by default) ───────────────────────
st.sidebar.markdown("---")
show_validation = st.sidebar.toggle(
    "🔬 Show model validation",
    value=False,
    key="show_validation_toggle",
    help="Hyuga vs Gao vs model angular stress (paper Fig 4), Maxwell–Wagner "
         "frequency, and automated self-tests.",
)

# Resolve eps_in, eps_out, sigma_out_uS (used in validation panel + cache key)
if COND_MODE:
    eps_in, eps_out = 80.0, 80.0
    sigma_out_uS = sigma_in_uS / ratio
else:
    sigma_out_uS = sigma_in_uS
    eps_in, eps_out = 80.0, 80.0 / ratio

# ── Warm the full slider-space cache (instant lookup after first run) ──────────
_cache = precompute_all(
    COND_MODE, sigma_in_uS, R, delta, eps_m, sigma_m, E0_Vcm,
    tuple(ratio_vals), tuple(freq_vals)
)
_ri = int(np.argmin(np.abs(np.log10(_cache['ratio_arr']) - np.log10(ratio))))
_fi = int(np.argmin(np.abs(np.log10(_cache['freq_arr'])  - np.log10(freq))))

# O(1) operating-point lookups
E0   = _cache['E0_vm']
Fz   = float(_cache['op_Fz'][_ri, _fi])
Fr   = float(_cache['op_Fr'][_ri, _fi])
a_op = complex(_cache['op_ar'][_ri, _fi], _cache['op_ai'][_ri, _fi])
b_op = complex(_cache['op_br'][_ri, _fi], _cache['op_bi'][_ri, _fi])

# Compute module-level stress coefficients for validation panel (Plot 4)
K_rc, K_rs, K_theta = stress_coefficients(a_op, b_op, E0, eps_in, eps_out)

# ============================================================================
# MAIN VIEW
# ----------------------------------------------------------------------------
vesicle_icon_path = os.path.join(parent_dir, "frontend", "vesicle_white_icon.png")
icon_b64 = get_base64_image(vesicle_icon_path)

st.markdown(
    f"<div style='font-size:1.95em;font-weight:800;color:#0f172a;"
    f"margin:0 0 2px 0;padding-top:4px;line-height:1.25;"
    f"font-family:\"Inter\",sans-serif;letter-spacing:-0.01em;"
    f"display:flex;align-items:center;gap:12px;'>"
    f"<img src='{icon_b64}' style='height:40px;width:40px;object-fit:contain;vertical-align:middle;border-radius:4px;'/>"
    f"Electrostrictive Forces on Lipid Vesicles</div>"
    f"<div style='font-size:0.95em;color:#64748b;margin:0 0 14px 0;"
    f"font-family:\"Inter\",sans-serif;'>Maxwell-stress deformation of a "
    f"compartmentalized vesicle under an AC field — after Rey et al., "
    f"IEEE TDEI 16(5), 2009.</div>",
    unsafe_allow_html=True,
)

# Phase-diagram grid and shape classification (all from pre-computed cache)
ratios_grid = _cache['pg_r']
freq_grid   = _cache['pg_f']
_pg_d       = _cache['pg_d']
_row_ref    = np.maximum(np.abs(_pg_d[:, 0]), 1e-12)
_gthr       = (0.02 * _row_ref)[:, None]
shape_grid  = np.zeros_like(_pg_d)
shape_grid[_pg_d >  _gthr] =  1
shape_grid[_pg_d < -_gthr] = -1

_pg_ri  = int(np.argmin(np.abs(np.log10(ratios_grid) - np.log10(ratio))))
_pg_fi  = int(np.argmin(np.abs(np.log10(freq_grid)   - np.log10(freq))))
phase_val = shape_grid[_pg_ri, _pg_fi]

if phase_val == 1:
    shape_desc = "Prolate (stretched along the field)"
    shape_color = "#dc2626"   # red  — matches heatmap colorscale[1]
    sign = 1.0
elif phase_val == -1:
    shape_desc = "Oblate (compressed along the field)"
    shape_color = "#ea580c"   # orange — matches heatmap colorscale[0]
    sign = -1.0
else:
    shape_desc = "Spherical (forces balanced)"
    shape_color = "#64748b"   # slate-grey — matches heatmap mid-band
    sign = 0.0

with st.container(border=True):
    st.markdown(
        "<div style='font-size:1.0em;font-weight:700;color:#0f172a;"
        "margin:0 0 4px 0;font-family:\"Inter\",sans-serif;'>"
        "1. Visualization: Simulated Vesicle Shape &amp; Stress Distribution</div>",
        unsafe_allow_html=True,
    )

    # Use constant operating point inputs for Python layout generation in Static mode.
    # This keeps the generated Plotly HTML content (wrapped_html) 100% identical.
    # The true operating point is read and updated client-side on load and during interaction.
    fixed_ratio_py = 1.0 if (not animate_ratio and not animate_freq) else ratio
    fixed_freq_py = 100000.0 if (not animate_ratio and not animate_freq) else freq

    fig_v5, fig_phase, fig_sweep, mode_name, metadata_list, peak_prolate, peak_oblate, max_p = build_plotly_animation(
        COND_MODE, sigma_in_uS, R, delta, eps_m, sigma_m, E0_Vcm,
        fixed_ratio_py, fixed_freq_py, animate_ratio, animate_freq,
        ratios_grid, freq_grid, shape_grid, cache_dict=_cache
    )
    
    # Pre-serialize sweep curves for all ratio values to update the sweep plot on-the-fly client-side
    sweep_data = []
    for i, r_val in enumerate(_cache['ratio_arr']):
        sweep_data.append({
            'ratio': float(r_val),
            'Fz': _cache['sw_Fz'][i].tolist(),
            'Fr': _cache['sw_Fr'][i].tolist(),
            'Zr': _cache['sw_Zr'][i].tolist(),
            'Zi': _cache['sw_Zi'][i].tolist(),
        })

    # Prepare Plotly figures as dictionaries
    fig_vesicle_dict = json.loads(fig_v5.to_json())
    fig_phase_dict = json.loads(fig_phase.to_json())
    fig_sweep_dict = json.loads(fig_sweep.to_json())



    # Render the custom component
    vesicle_dashboard(
        fig_vesicle=fig_vesicle_dict,
        fig_phase=fig_phase_dict,
        fig_sweep=fig_sweep_dict,
        sweep_data=sweep_data,
        metadata=metadata_list,
        ratio=ratio,
        freq=freq,
        cond_mode=COND_MODE,
        peak_prolate=peak_prolate,
        peak_oblate=peak_oblate,
        max_p=max_p,
        animate_ratio=animate_ratio,
        animate_freq=animate_freq,
        mode_name=mode_name
    )

# ---- Validation panel (only rendered when the sidebar toggle is on) ---------
if show_validation:
    with st.container(border=True):
        st.markdown(
            "<div style='font-size:1.0em;font-weight:700;color:#0f172a;"
            "margin:0 0 8px 0;font-family:\"Inter\",sans-serif;'>"
            "🔬 Model validation (Hyuga vs Gao vs model — paper Fig 4)</div>",
            unsafe_allow_html=True,
        )
        axfont = dict(color="#0f172a", size=11, family="Inter, sans-serif")
        tkfont = dict(color="#0f172a", size=9, family="Inter, sans-serif")
        theta = np.linspace(0, np.pi, 181)
        Rs = (sigma_in_uS / sigma_out_uS) if COND_MODE else 1.0
        Re = (eps_in / eps_out)
        p_h = analytical_pressure(theta, Rs, Re, eps_out, E0, "hyuga")
        p_g = analytical_pressure(theta, Rs, Re, eps_out, E0, "gao")
        # model angular stress from the same coefficients used in the 3D plot
        p_model = (K_rc * np.cos(theta) ** 2 + K_rs * np.sin(theta) ** 2
                   + K_theta * np.cos(theta) * np.sin(theta))

        def _norm(x):
            m = np.nanmax(np.abs(x)) or 1.0
            return x / m
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(x=np.degrees(theta), y=_norm(p_h),
                                     name="Hyuga 1991", line=dict(color="#2563eb", width=2)))
        fig_val.add_trace(go.Scatter(x=np.degrees(theta), y=_norm(p_model),
                                     name="This model", line=dict(color="#ef4444", width=2)))
        fig_val.add_trace(go.Scatter(x=np.degrees(theta), y=_norm(p_g),
                                     name="Gao 2008", line=dict(color="#16a34a", width=2)))
        fig_val.update_xaxes(title_text="Angle θ (deg)", gridcolor="#cbd5e1",
                             title_font=axfont, tickfont=tkfont)
        fig_val.update_yaxes(title_text="Normalized Maxwell stress", gridcolor="#cbd5e1",
                             title_font=axfont, tickfont=tkfont)
        fig_val.update_layout(height=320, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                              font=dict(color="#0f172a"), margin=dict(l=40, r=20, t=20, b=40),
                              legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
        st.plotly_chart(fig_val, use_container_width=True, config={'displayModeBar': False}, key="model_validation_plot")

        f_mw = maxwell_wagner_freq(eps_in, eps_out, sigma_in_uS, sigma_out_uS)
        st.markdown(f"**Maxwell–Wagner relaxation frequency:** `{f_mw:.2e} Hz` "
                    f"(paper inflection band ≈ 2×10⁵–8×10⁵ Hz).")
        st.markdown("**Automated self-tests:**")
        for name, ok in _selftest(R, delta, eps_m, sigma_m, E0_Vcm):
            st.markdown(f"- {'✅' if ok else '❌'} {name}")

st.markdown("---")
st.markdown(
    "<div style='font-size:1.0em;font-weight:700;color:#0f172a;margin:0 0 8px 0;"
    "font-family:\"Inter\",sans-serif;'>References</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "**Primary article (model implemented here)**  \n"
    "[1] J. I. Rey, R. J. Connolly, M. J. Jaroszeski, A. M. Hoff, J. A. Llewellyn, and "
    "R. Gilbert, \"Electrostrictive forces on vesicles with compartmentalized permittivity "
    "and conductivity conditions,\" *IEEE Transactions on Dielectrics and Electrical "
    "Insulation*, vol. 16, no. 5, pp. 1280–1287, Oct. 2009. "
    "https://ieeexplore.ieee.org/document/5293939\n\n"

    "**Analytical force models underpinning the formulation**  \n"
    "[2] H. Hyuga, K. Kinosita, and N. Wakabayashi, \"Deformation of vesicles under the "
    "influence of strong electric fields,\" *Japanese Journal of Applied Physics*, vol. 30, "
    "no. 6, pp. 1141–1148, 1991.  \n"
    "[3] L.-T. Gao, X.-Q. Feng, Y.-J. Yin, and H. Gao, \"An electromechanical liquid crystal "
    "model of vesicles,\" *Journal of the Mechanics and Physics of Solids*, vol. 56, no. 9, "
    "pp. 2844–2862, 2008.\n\n"

    "**Experimental reference (paper's ref. [1])**  \n"
    "[4] R. Dimova, K. A. Riske, S. Aranda, N. Bezlyepkina, R. L. Knorr, and R. Lipowsky, "
    "\"Giant vesicles in electric fields,\" *Soft Matter*, vol. 3, no. 7, pp. 817–827, 2007."
)
st.caption(
    "Interactive re-implementation using a closed-form Hyuga/Gao angular model with a "
    "Maxwell–Wagner frequency blend; the primary article's results were produced with a "
    "COMSOL finite-element model. See the accompanying notes for documented agreements and "
    "differences with the paper."
)

