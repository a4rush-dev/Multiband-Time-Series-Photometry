import os

# Make RadVel happy on Windows
if "HOME" not in os.environ:
    os.environ["HOME"] = os.environ.get("USERPROFILE", os.path.expanduser("~"))

import numpy as np
import matplotlib.pyplot as plt
import batman
import emcee
import radvel
from radvel import fitting
import radvel.likelihood as rlike
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

# RV IMPORT HERE

PERIOD   = 5.6334729
TC_REF   = 2454529.674
E_LIT    = 0.502
WDEG_LIT = 188.8
WRAD_LIT = np.deg2rad(WDEG_LIT)

SECOSW_LIT = np.sqrt(E_LIT) * np.cos(WRAD_LIT)
SESINW_LIT = np.sqrt(E_LIT) * np.sin(WRAD_LIT)

SIG_ORB_OFFSET = 5e-4   # prior width for per-HST-orbit additive offsets

rv_model = HatP2RVModel(rv_file="hat_p2_rv.txt")

t_rv = rv_model.t_all
rv_all = rv_model.rv_all
err_rv = rv_model.err_all
inst_id = rv_model.inst_id
mask_hires = rv_model.mask_hires
mask_harpsn = rv_model.mask_harpsn
time_base = rv_model.time_base
t_rel_rv = rv_model.t_rel

theta_rv_init = rv_model.theta_from_params()
tc1_init = theta_rv_init[0]
k1_init = theta_rv_init[1]

print("\nRV-only initialization from modular RV model:")
print(f"  tc1_init = {tc1_init:.6f}")
print(f"  k1_init  = {k1_init:.3f}")

INPUT_TXT = "full_LC_stel_puls_orb_params.txt"
hst_data = np.loadtxt(INPUT_TXT)

time_hst_bjd = hst_data[0, :]
flux_hst = hst_data[1, :]
ferr_hst = hst_data[2, :]

t_hst_rel = time_hst_bjd - time_hst_bjd[0]
flux_hst_norm = flux_hst + 1.0
tmid_hst = np.median(t_hst_rel)

print("HST visit:")
print(f"  N points   = {len(time_hst_bjd)}")
print(f"  BJD range  = {time_hst_bjd.min():.8f} to {time_hst_bjd.max():.8f}")
print(f"  duration   = {t_hst_rel[-1]:.8f} days")
print(f"  median dt  = {np.median(np.diff(time_hst_bjd)):.4e} days")

# Optional error inflation
oot_mask = (t_hst_rel > 0.3)
if np.sum(oot_mask) > 5:
    out_of_transit_std = np.std(flux_hst_norm[oot_mask])
else:
    out_of_transit_std = np.std(flux_hst_norm)

print(f"median HST flux_err: {np.median(ferr_hst):.2e}")
print(f"HST std outside early visit: {out_of_transit_std:.2e}")

if out_of_transit_std > 3 * np.median(ferr_hst):
    print("WARNING: HST flux scatter >> quoted flux_err, inflating uncertainties")
    ferr_hst = np.sqrt(ferr_hst**2 + (0.5 * out_of_transit_std)**2)

dt_hst = np.diff(t_hst_rel)
gap_thresh = max(5.0 * np.median(dt_hst), 0.015)

orbit_breaks = np.where(dt_hst > gap_thresh)[0] + 1
orbit_starts = np.r_[0, orbit_breaks]
orbit_ends = np.r_[orbit_breaks, len(t_hst_rel)]

orbit_id = np.zeros(len(t_hst_rel), dtype=int)
for k, (s, e) in enumerate(zip(orbit_starts, orbit_ends)):
    orbit_id[s:e] = k

n_hst_orbits = len(orbit_starts)

print(f"\nIdentified {n_hst_orbits} HST orbit chunks")
print(f"gap threshold = {gap_thresh:.5f} days")

t0_rel_guess = 0.14
t0_hst_bjd_guess = time_hst_bjd[0] + t0_rel_guess
N_hst = int(np.round((t0_hst_bjd_guess - tc1_init) / PERIOD))

print(f"\nNearest HST orbit count from RV tc1: N_hst = {N_hst}")

base_params = batman.TransitParams()
base_params.per = PERIOD
base_params.ecc = E_LIT
base_params.w = WDEG_LIT
base_params.limb_dark = "quadratic"

def planet_flux_model(t_rel, Fp_Fsmin, c1, c2, c3, c4, c5, c6):
    u = np.where(t_rel < c2, (t_rel - c2) / c3, (t_rel - c2) / c4)
    eclipse = (t_rel > (c5 - 0.5 * c6)) & (t_rel < (c5 + 0.5 * c6))
    model = np.where(eclipse, 0.0, Fp_Fsmin + c1 / (u**2 + 1.0))
    return model

def theta_initial():
    return np.array([
        theta_rv_init[0],             # tc1
        theta_rv_init[1],             # k1
        theta_rv_init[2],             # secosw1
        theta_rv_init[3],             # sesinw1
        theta_rv_init[4],             # gamma_hires
        theta_rv_init[5],             # gamma_harpsn
        theta_rv_init[6],             # ln_jit_hires
        theta_rv_init[7],             # ln_jit_harpsn
        0.14,                         # t0_rel
        0.0,                          # h0
        0.0,                          # h1
        0.070,                        # rp
        8.9,                          # a/R*
        86.3,                         # inc
        0.2,                          # u1
        0.3,                          # u2
        5e-5,                         # Fp_Fsmin
        5e-4,                         # c1
        1.05,                         # c2
        1.5,                          # c3
        0.2,                          # c4
        1.23,                         # c5
        0.11,                         # c6
        np.log(np.median(ferr_hst)),  # ln_jit_hst
    ], dtype=float)

RV_NDIM = 8  # number of RV parameters in theta

def split_theta(theta):
    theta = np.asarray(theta)
    theta_rv = theta[:RV_NDIM]
    theta_hst = theta[RV_NDIM:]
    return theta_rv, theta_hst

def theta_to_params(theta):
    """
    Update RV model parameters and BATMAN base_params from theta.
    Returns (e_val, omega_val).
    """
    theta_rv, theta_hst = split_theta(theta)

    rv_model.update_params_from_theta(theta_rv)

    (
        t0_rel,
        h0, h1,
        rp, a, inc, u1, u2,
        Fp_Fsmin, c1, c2, c3, c4, c5, c6,
        ln_jit_hst,
    ) = theta_hst

    secosw1 = theta_rv[2]
    sesinw1 = theta_rv[3]
    e_val, omega_val = e_omega_from_se(secosw1, sesinw1)

    base_params.ecc = e_val
    base_params.w = np.degrees(omega_val)
    base_params.rp = rp
    base_params.a = a
    base_params.inc = inc
    base_params.u = [u1, u2]

    return e_val, omega_val

def hst_base_model(theta, t_rel_arr):
    theta_rv, theta_hst = split_theta(theta)

    (
        t0_rel,
        h0, h1,
        rp, a, inc, u1, u2,
        Fp_Fsmin, c1, c2, c3, c4, c5, c6,
        ln_jit_hst,
    ) = theta_hst

    theta_to_params(theta)

    base_params.t0 = t0_rel

    m = batman.TransitModel(
        base_params, t_rel_arr,
        supersample_factor=3,
        exp_time=0.002,
    )
    transit = m.light_curve(base_params)
    planet = planet_flux_model(t_rel_arr, Fp_Fsmin, c1, c2, c3, c4, c5, c6)
    visit_baseline = h0 + h1 * (t_rel_arr - tmid_hst)

    return transit + planet - Fp_Fsmin + visit_baseline

def hst_full_model_and_offsets(theta):
    base_model = hst_base_model(theta, t_hst_rel)

    theta_rv, theta_hst = split_theta(theta)
    ln_jit_hst = theta_hst[-1]
    jit_hst = np.exp(ln_jit_hst)
    sigma_hst = np.sqrt(ferr_hst**2 + jit_hst**2)

    model = base_model.copy()
    orbit_offsets = np.zeros(n_hst_orbits)

    for k in range(n_hst_orbits):
        m = (orbit_id == k)
        w = 1.0 / sigma_hst[m]**2
        resid = flux_hst_norm[m] - model[m]

        denom = np.sum(w) + 1.0 / SIG_ORB_OFFSET**2
        numer = np.sum(w * resid)

        orbit_offsets[k] = numer / denom
        model[m] += orbit_offsets[k]

    ln_prior_offsets = -0.5 * np.sum(
        (orbit_offsets / SIG_ORB_OFFSET)**2
        + np.log(2 * np.pi * SIG_ORB_OFFSET**2)
    )

    return model, sigma_hst, orbit_offsets, ln_prior_offsets

def rv_log_likelihood(theta):
    theta_rv, theta_hst = split_theta(theta)
    return rv_model.log_likelihood(theta_rv)

def hst_log_likelihood(theta):
    model_flux, sigma_hst, orbit_offsets, ln_prior_offsets = hst_full_model_and_offsets(theta)
    resid = flux_hst_norm - model_flux

    lnL_data = -0.5 * np.sum((resid / sigma_hst)**2 + np.log(2 * np.pi * sigma_hst**2))
    return lnL_data + ln_prior_offsets

def log_prior(theta):
    theta_rv, theta_hst = split_theta(theta)

    # RV priors from modular RV model
    lp_rv = rv_model.log_prior(theta_rv)
    if not np.isfinite(lp_rv):
        return -np.inf

    (
        t0_rel,
        h0, h1,
        rp, a, inc, u1, u2,
        Fp_Fsmin, c1, c2, c3, c4, c5, c6,
        ln_jit_hst,
    ) = theta_hst

    # HST box priors
    if not (0.0 < t0_rel < 0.3):
        return -np.inf
    if not (-2e-3 < h0 < 2e-3):
        return -np.inf
    if not (-5e-3 < h1 < 5e-3):
        return -np.inf

    if not (0.05 < rp < 0.09):
        return -np.inf
    if not (7.0 < a < 11.0):
        return -np.inf
    if not (84.0 < inc < 89.0):
        return -np.inf
    if not (0.0 < u1 < 1.0):
        return -np.inf
    if not (0.0 < u2 < 1.0):
        return -np.inf
    if not (-1e-4 < Fp_Fsmin < 5e-4):
        return -np.inf
    if not (1e-4 < c1 < 1.5e-3):
        return -np.inf
    if not (0.8 < c2 < 1.2):
        return -np.inf
    if not (0.5 < c3 < 4.0):
        return -np.inf
    if not (0.1 < c4 < 0.5):
        return -np.inf
    if not (1.15 < c5 < 1.35):
        return -np.inf
    if not (0.05 < c6 < 0.2):
        return -np.inf
    if not (np.log(1e-6) < ln_jit_hst < np.log(5e-3)):
        return -np.inf

    lp = lp_rv

    # Gaussian priors for HST baseline terms
    sig_h0 = 3e-4
    sig_h1 = 1e-3
    lp += -0.5 * (h0**2 / sig_h0**2 + np.log(2 * np.pi * sig_h0**2))
    lp += -0.5 * (h1**2 / sig_h1**2 + np.log(2 * np.pi * sig_h1**2))

    # Soft RV-HST timing link
    tc1 = theta_rv[0]
    t_trans_hst_bjd = time_hst_bjd[0] + t0_rel
    t_trans_rv_pred = tc1 + N_hst * PERIOD
    dt_link = t_trans_hst_bjd - t_trans_rv_pred

    sig_link = 0.12
    lp += -0.5 * (dt_link**2 / sig_link**2 + np.log(2 * np.pi * sig_link**2))

    return lp

def log_prob(theta, alpha_hst=1.0):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    lnL_rv = rv_log_likelihood(theta)
    if not np.isfinite(lnL_rv):
        return -np.inf

    lnL_hst = hst_log_likelihood(theta)
    if not np.isfinite(lnL_hst):
        return -np.inf

    return lp + lnL_rv + alpha_hst * lnL_hst

theta_start = theta_initial()
ndim = len(theta_start)
nwalkers = 64

scatter = np.array([
    1e-3,   # tc1
    5.0,    # k1
    1e-3,   # secosw1
    1e-3,   # sesinw1
    5.0,    # gamma_hires
    5.0,    # gamma_harpsn
    0.1,    # ln_jit_hires
    0.1,    # ln_jit_harpsn
    1e-3,   # t0_rel
    5e-5,   # h0
    5e-5,   # h1
    1e-3,   # rp
    1e-2,   # a
    0.1,    # inc
    1e-3,   # u1
    1e-3,   # u2
    1e-5,   # Fp_Fsmin
    1e-5,   # c1
    1e-3,   # c2
    1e-3,   # c3
    1e-3,   # c4
    1e-3,   # c5
    1e-3,   # c6
    0.1,    # ln_jit_hst
])
assert scatter.shape[0] == ndim

pos0 = theta_start + scatter * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_prob, kwargs={"alpha_hst": 1.0}
)

print("\nJOINT RV+HST: burn-in...")
sampler.run_mcmc(pos0, 3000, progress=True)
sampler.reset()
print("JOINT RV+HST: production...")
sampler.run_mcmc(None, 6000, progress=True)

flat = sampler.get_chain(thin=10, flat=True)
theta_med = np.percentile(flat, 50, axis=0)

theta_rv_med, theta_hst_med = split_theta(theta_med)

(
    tc1_med, k1_med, secosw_med, sesinw_med,
    gamma_hires_med, gamma_harpsn_med,
    ln_jit_hires_med, ln_jit_harpsn_med,
) = theta_rv_med

(
    t0_rel_med,
    h0_med, h1_med,
    rp_med, a_med, inc_med, u1_med, u2_med,
    Fp_Fsmin_med, c1_med, c2_med, c3_med, c4_med, c5_med, c6_med,
    ln_jit_hst_med,
) = theta_hst_med

jit_hires_med = np.exp(ln_jit_hires_med)
jit_harpsn_med = np.exp(ln_jit_harpsn_med)
jit_hst_med = np.exp(ln_jit_hst_med)
e_med, omega_med = e_omega_from_se(secosw_med, sesinw_med)

theta_to_params(theta_med)

rv_diag = rv_model.compute_diagnostics(theta_rv_med)

print("\nJOINT RV+HST DIAGNOSTICS")
print("========================")
print(f"tc1               = {tc1_med:.6f}")
print(f"k1                = {k1_med:.3f} m/s")
print(f"e                 = {e_med:.4f}")
print(f"omega (deg)       = {np.rad2deg(omega_med):.2f}")
print(f"gamma_hires       = {gamma_hires_med:.3f} m/s")
print(f"gamma_harpsn      = {gamma_harpsn_med:.3f} m/s")
print(f"jit_hires         = {jit_hires_med:.3f} m/s")
print(f"jit_harpsn        = {jit_harpsn_med:.3f} m/s")
print(f"N_hst             = {N_hst}")
print(f"t0_rel            = {t0_rel_med:.6f} days")
print(f"h0                = {h0_med:.6e}")
print(f"h1                = {h1_med:.6e} per day")
print(f"jit_hst           = {jit_hst_med:.6e}")
print(f"rp                = {rp_med:.6f}")
print(f"a (a/R*)          = {a_med:.6f}")
print(f"inc               = {inc_med:.3f}")
print(f"u1                = {u1_med:.3f}")
print(f"u2                = {u2_med:.3f}")
print(f"Fp_Fsmin          = {Fp_Fsmin_med:.6e}")
print(f"c1                = {c1_med:.6e}")
print(f"c2                = {c2_med:.6f}")
print(f"c3                = {c3_med:.6f}")
print(f"c4                = {c4_med:.6f}")
print(f"c5                = {c5_med:.6f}")
print(f"c6                = {c6_med:.6f}")
print(f"RV chi2           = {rv_diag['chi2']:.2f}  (reduced ~ {rv_diag['chi2r']:.3f})")
print(f"RV RMS            = {rv_diag['rms']:.2f} m/s")

best_hst_model, sigma_hst_med, orbit_offsets_med, ln_prior_offsets_med = hst_full_model_and_offsets(theta_med)
hst_residuals = flux_hst_norm - best_hst_model

chi2_hst = np.sum((hst_residuals / sigma_hst_med)**2)
dof_hst = len(flux_hst_norm) - (len(theta_hst_med) + n_hst_orbits)
chi2r_hst = chi2_hst / dof_hst

res_skew = skew(hst_residuals / sigma_hst_med)
res_kurt = kurtosis(hst_residuals / sigma_hst_med)

t_trans_hst_bjd_med = time_hst_bjd[0] + t0_rel_med
t_trans_rv_pred_med = tc1_med + N_hst * PERIOD
dt_link_med = t_trans_hst_bjd_med - t_trans_rv_pred_med

print(f"chi2_hst          = {chi2_hst:.2f} (reduced ~ {chi2r_hst:.3f})")
print(f"HST skew          = {res_skew:.3f}")
print(f"HST kurtosis      = {res_kurt:.3f}")
print(f"HST transit BJD   = {t_trans_hst_bjd_med:.6f}")
print(f"RV-pred BJD       = {t_trans_rv_pred_med:.6f}")
print(f"timing offset     = {dt_link_med:.6f} days")
print(f"RMS orbit offset  = {np.std(orbit_offsets_med):.6e}")

rv_model.plot_phase_rv(theta_rv=theta_rv_med)
rv_model.plot_residual_histogram(theta_rv=theta_rv_med)

# HST data + model
fig_hst, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

ax1.errorbar(
    t_hst_rel,
    flux_hst_norm,
    yerr=ferr_hst,
    fmt=".k",
    ms=3,
    alpha=0.4,
    label="HST data",
)
ax1.plot(
    t_hst_rel,
    best_hst_model,
    "r-",
    lw=2,
    label="joint RV+HST best fit",
)
ax1.set_ylabel("Normalized flux")
ax1.legend()
ax1.set_title(
    f"HST reduced χ² = {chi2r_hst:.3f} | skew = {res_skew:.3f} | kurtosis = {res_kurt:.3f}"
)

ax2.errorbar(
    t_hst_rel,
    hst_residuals,
    yerr=ferr_hst,
    fmt=".k",
    ms=3,
    alpha=0.5,
)
ax2.axhline(0, color="r", linestyle="--", lw=1)
ax2.set_xlabel("Time since HST visit start (days)")
ax2.set_ylabel("Residuals")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.axhline(0, color="k", lw=1)
plt.plot(np.arange(n_hst_orbits), orbit_offsets_med, "o-")
plt.xlabel("HST orbit index")
plt.ylabel("Recovered additive offset")
plt.title("Recovered per-orbit HST offsets")
plt.tight_layout()
plt.show()