import os


# Make RadVel happy on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', os.path.expanduser('~'))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batman
import emcee
import radvel
from radvel import fitting
import radvel.likelihood as rlike
import warnings
warnings.filterwarnings('ignore')

PERIOD   = 5.6334729       # HAT-P-2b period (days)
TC_REF   = 2454529.674     # reference transit epoch (BJD)
E_LIT    = 0.502
WDEG_LIT = 188.8
WRAD_LIT = np.deg2rad(WDEG_LIT)

SECOSW_LIT = np.sqrt(E_LIT) * np.cos(WRAD_LIT)
SESINW_LIT = np.sqrt(E_LIT) * np.sin(WRAD_LIT)

LN_JIT_CENTER = np.log(5.0)   # RV jitter prior center (m/s)
LN_JIT_SIGMA  = 0.7           # RV jitter prior width


def e_omega_from_se(secosw_val, sesinw_val):
    e_val = secosw_val**2 + sesinw_val**2
    omega_val = np.arctan2(sesinw_val, secosw_val)
    return e_val, omega_val


def phase_fold(t, tc_use, per_use):
    return np.mod((t - tc_use) / per_use, 1.0)


rv_data = np.loadtxt("hat_p2_rv.txt")
t_rv    = rv_data[:, 0] + 2450000.0
rv_all  = rv_data[:, 1]
err_rv  = rv_data[:, 2]
inst_id = rv_data[:, 3].astype(int)

mask_hires  = (inst_id == 0)
mask_harpsn = (inst_id == 1)

time_base = np.median(t_rv)
t_rel_rv  = t_rv - time_base  # for secular trend g1,g2


df_tess = pd.read_csv("hatp2_tess_lightcurve.csv")
t_tess_btjd = np.array(df_tess["time"], dtype=float)        # BTJD
f_tess      = np.array(df_tess["flux"], dtype=float)
ferr_tess   = np.array(df_tess["flux_err"], dtype=float)

# basic sanity check on flux errors
if np.any(~np.isfinite(ferr_tess)) or np.any(ferr_tess <= 0):
    ferr_tess = np.ones_like(f_tess) * np.std(f_tess)
    print("warning: flux_err had bad values -> replaced with global scatter estimate")

# normalize flux
median_flux = np.median(f_tess)
f_tess    = f_tess / median_flux
ferr_tess = ferr_tess / median_flux

# convert BTJD -> BJD
TESS_BTJD_OFFSET = 2457000.0
t_tess_bjd = t_tess_btjd + TESS_BTJD_OFFSET

# quick grid search to locate a TESS transit epoch (in BJD)
ntrial = 1200
trial_offsets = np.linspace(np.nanmin(t_tess_bjd),
                            np.nanmin(t_tess_bjd) + PERIOD,
                            ntrial)
window_phase = 0.02
depths = np.zeros(ntrial)

for i, off in enumerate(trial_offsets):
    phase = ((t_tess_bjd - off) / PERIOD) % 1.0
    phase = phase - np.round(phase)
    mask = np.abs(phase) < window_phase
    if np.sum(mask) < 5:
        depths[i] = np.nan
    else:
        depths[i] = np.nanmedian(f_tess[mask])

best_idx = np.nanargmin(depths)
t0_tess_bjd_est = trial_offsets[best_idx]
print("TESS t0 (grid-search, BJD) =", t0_tess_bjd_est)

# window around that TESS transit
phase_tess = ((t_tess_bjd - t0_tess_bjd_est) / PERIOD) % 1.0
phase_tess = phase_tess - np.round(phase_tess)
window = 0.05
mask_window = np.abs(phase_tess) < window

t_win_bjd  = t_tess_bjd[mask_window]
f_win      = f_tess[mask_window]
ferr_win   = ferr_tess[mask_window]

order = np.argsort(t_win_bjd)
t_win_bjd  = t_win_bjd[order]
f_win      = f_win[order]
ferr_win   = ferr_win[order]

params_static = radvel.Parameters(1, basis="per tc secosw sesinw k")
params_static['per1']    = radvel.Parameter(value=PERIOD,    vary=False)
params_static['tc1']     = radvel.Parameter(value=TC_REF,    vary=True)
params_static['secosw1'] = radvel.Parameter(value=SECOSW_LIT, vary=False)
params_static['sesinw1'] = radvel.Parameter(value=SESINW_LIT, vary=False)
params_static['k1']      = radvel.Parameter(value=950.0,      vary=True)

params_static['gamma_hires']  = radvel.Parameter(value=np.mean(rv_all[mask_hires]),
                                                 vary=True, linear=True)
params_static['jit_hires']    = radvel.Parameter(value=3.0, vary=True)
params_static['gamma_harpsn'] = radvel.Parameter(value=np.mean(rv_all[mask_harpsn]),
                                                 vary=True, linear=True)
params_static['jit_harpsn']   = radvel.Parameter(value=3.0, vary=True)

mod_static = radvel.RVModel(params_static)
mod_static.time_base = time_base

like_hires = rlike.RVLikelihood(
    mod_static, t_rv[mask_hires], rv_all[mask_hires], err_rv[mask_hires],
    suffix="_hires"
)
like_harpsn = rlike.RVLikelihood(
    mod_static, t_rv[mask_harpsn], rv_all[mask_harpsn], err_rv[mask_harpsn],
    suffix="_harpsn"
)

like_hires.params['gamma_hires']   = params_static['gamma_hires']
like_hires.params['jit_hires']     = params_static['jit_hires']
like_harpsn.params['gamma_harpsn'] = params_static['gamma_harpsn']
like_harpsn.params['jit_harpsn']   = params_static['jit_harpsn']

comp_like = rlike.CompositeLikelihood([like_hires, like_harpsn])
comp_like = fitting.maxlike_fitting(comp_like)
best_static = comp_like.params

tc1_init = best_static['tc1'].value
k1_init  = best_static['k1'].value
print("RV-only static tc1 =", tc1_init, "k1 =", k1_init)

N_cycles = int(np.round((t0_tess_bjd_est - tc1_init) / PERIOD))
print("Integer cycle count N between RV tc1 and TESS t0:", N_cycles)


# JOINT MODEL PARAMETERS (RV+TESS)
# theta = [
#   tc1, k1, secosw1, sesinw1,
#   gamma_hires, gamma_harpsn,
#   ln_jit_hires, ln_jit_harpsn,
#   g1, g2,
#   delta_t, ln_jit_tess,
#   rp, a, inc
# ]
params_joint = radvel.Parameters(1, basis="per tc secosw sesinw k")
params_joint['per1']    = radvel.Parameter(value=PERIOD,    vary=False)
params_joint['tc1']     = radvel.Parameter(value=tc1_init,  vary=True)
params_joint['secosw1'] = radvel.Parameter(value=SECOSW_LIT, vary=True)
params_joint['sesinw1'] = radvel.Parameter(value=SESINW_LIT, vary=True)
params_joint['k1']      = radvel.Parameter(value=k1_init,    vary=True)

params_joint['gamma_hires']  = radvel.Parameter(value=best_static['gamma_hires'].value,
                                                vary=True, linear=True)
params_joint['jit_hires']    = radvel.Parameter(value=best_static['jit_hires'].value, vary=True)
params_joint['gamma_harpsn'] = radvel.Parameter(value=best_static['gamma_harpsn'].value,
                                                vary=True, linear=True)
params_joint['jit_harpsn']   = radvel.Parameter(value=best_static['jit_harpsn'].value, vary=True)

params_joint['g1'] = radvel.Parameter(value=0.0, vary=True, linear=True)
params_joint['g2'] = radvel.Parameter(value=0.0, vary=True, linear=True)

mod_joint = radvel.RVModel(params_joint)
mod_joint.time_base = time_base

# BATMAN params (TESS) will be updated from theta
bat_params = batman.TransitParams()
bat_params.t0 = 0.0
bat_params.per = PERIOD
bat_params.rp = 0.0722
bat_params.a = 8.9
bat_params.inc = 86.3
bat_params.ecc = E_LIT
bat_params.w = WDEG_LIT
bat_params.limb_dark = "quadratic"
bat_params.u = [0.2, 0.3]


def theta_initial():
    tc1      = params_joint['tc1'].value
    k1       = params_joint['k1'].value
    secosw1  = params_joint['secosw1'].value
    sesinw1  = params_joint['sesinw1'].value
    gamma_h  = params_joint['gamma_hires'].value
    gamma_hp = params_joint['gamma_harpsn'].value
    ln_jit_h = np.log(params_joint['jit_hires'].value)
    ln_jit_hp = np.log(params_joint['jit_harpsn'].value)
    g1       = params_joint['g1'].value
    g2       = params_joint['g2'].value

    # TESS initial values from TESS-only fit idea
    delta_t_init    = t0_tess_bjd_est - (tc1 + N_cycles * PERIOD)  # small offset, ~0
    ln_jit_tess_init = np.log(0.001)   # jitter in flux units; tune as needed
    rp_init         = bat_params.rp
    a_init          = bat_params.a
    inc_init        = bat_params.inc

    return np.array([
        tc1, k1, secosw1, sesinw1,
        gamma_h, gamma_hp,
        ln_jit_h, ln_jit_hp,
        g1, g2,
        delta_t_init, ln_jit_tess_init,
        rp_init, a_init, inc_init
    ], dtype=float)


def theta_to_params(theta):
    (tc1, k1, secosw1, sesinw1,
     gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn,
     g1, g2,
     delta_t, ln_jit_tess,
     rp, a, inc) = theta

    # RV parameters
    params_joint['tc1'].value          = tc1
    params_joint['k1'].value           = k1
    params_joint['secosw1'].value      = secosw1
    params_joint['sesinw1'].value      = sesinw1
    params_joint['gamma_hires'].value  = gamma_hires
    params_joint['gamma_harpsn'].value = gamma_harpsn
    params_joint['jit_hires'].value    = np.exp(ln_jit_hires)
    params_joint['jit_harpsn'].value   = np.exp(ln_jit_harpsn)
    params_joint['g1'].value           = g1
    params_joint['g2'].value           = g2

    # shared eccentric orbit for TESS
    e_val, omega_val = e_omega_from_se(secosw1, sesinw1)
    bat_params.ecc = e_val
    bat_params.w   = np.degrees(omega_val)

    # TESS geometry + local epoch offset
    bat_params.rp  = rp
    bat_params.a   = a
    bat_params.inc = inc

    # store delta_t and ln_jit_tess in module-level variables for TESS likelihood
    return delta_t, ln_jit_tess


def rv_log_likelihood(theta):
    delta_t, ln_jit_tess = theta_to_params(theta)
    rv_planet = mod_joint(t_rv)

    (tc1, k1, secosw1, sesinw1,
     gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn,
     g1, g2,
     delta_t, ln_jit_tess,
     rp, a, inc) = theta

    jit_hires  = np.exp(ln_jit_hires)
    jit_harpsn = np.exp(ln_jit_harpsn)

    rv_model = rv_planet.copy()
    rv_model[mask_hires]  += gamma_hires
    rv_model[mask_harpsn] += gamma_harpsn
    rv_model += g1 * t_rel_rv + g2 * (t_rel_rv**2)

    sigma_hires  = np.sqrt(err_rv[mask_hires]**2  + jit_hires**2)
    sigma_harpsn = np.sqrt(err_rv[mask_harpsn]**2 + jit_harpsn**2)

    resid_hires  = rv_all[mask_hires]  - rv_model[mask_hires]
    resid_harpsn = rv_all[mask_harpsn] - rv_model[mask_harpsn]

    lnL_hires  = -0.5 * np.sum((resid_hires / sigma_hires)**2
                               + np.log(2*np.pi*sigma_hires**2))
    lnL_harpsn = -0.5 * np.sum((resid_harpsn / sigma_harpsn)**2
                               + np.log(2*np.pi*sigma_harpsn**2))
    return lnL_hires + lnL_harpsn


def tess_model(theta, t_bjd_window):
    (tc1, k1, secosw1, sesinw1,
     gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn,
     g1, g2,
     delta_t, ln_jit_tess,
     rp, a, inc) = theta

    # joint transit epoch in BJD: RV tc1 + N*P + delta_t
    t0_bjd_joint = tc1 + N_cycles * PERIOD + delta_t

    # relative times for BATMAN
    t_rel = t_bjd_window - t0_bjd_joint

    bat_params.t0 = 0.0  # transit at t_rel = 0
    bat_params.rp = rp
    bat_params.a  = a
    bat_params.inc = inc

    # use median cadence as exposure time
    if len(t_rel) > 1:
        exp_time = np.median(np.diff(t_rel))
    else:
        exp_time = 0.002

    m = batman.TransitModel(
        bat_params, t_rel,
        supersample_factor=5,
        exp_time=exp_time
    )
    return m.light_curve(bat_params)

def tess_log_likelihood(theta):
    delta_t, ln_jit_tess = theta_to_params(theta)

    jit_tess = np.exp(ln_jit_tess)
    flux_model = tess_model(theta, t_win_bjd)

    sigma_tess = np.sqrt(ferr_win**2 + jit_tess**2)
    resid = (f_win - flux_model) / sigma_tess

    lnL_tess = -0.5 * np.sum(resid**2 + np.log(2*np.pi*sigma_tess**2))
    return lnL_tess

def log_prior(theta):
    (tc1, k1, secosw1, sesinw1,
     gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn,
     g1, g2,
     delta_t, ln_jit_tess,
     rp, a, inc) = theta

    # RV parameter bounds
    if not (TC_REF - 0.5 < tc1 < TC_REF + 0.5):
        return -np.inf
    if not (500.0 < k1 < 1500.0):
        return -np.inf
    if not (-2000.0 < gamma_hires < 2000.0):
        return -np.inf
    if not (-2000.0 < gamma_harpsn < 2000.0):
        return -np.inf
    if not (np.log(0.5) < ln_jit_hires < np.log(30.0)):
        return -np.inf
    if not (np.log(0.5) < ln_jit_harpsn < np.log(30.0)):
        return -np.inf
    if not (-1.0 < g1 < 1.0):
        return -np.inf
    if not (-1e-3 < g2 < 1e-3):
        return -np.inf

    # eccentricity bounds
    e_val, omega_val = e_omega_from_se(secosw1, sesinw1)
    if not (0.0 < e_val < 0.999):
        return -np.inf

    # TESS-related bounds
    if not (-0.5 < delta_t < 0.5):
        return -np.inf
    if not (np.log(1e-5) < ln_jit_tess < np.log(0.1)):
        return -np.inf
    if not (0.04 < rp < 0.11):
        return -np.inf
    if not (5.0 < a < 15.0):
        return -np.inf
    if not (80.0 < inc < 90.0):
        return -np.inf

    lp = 0.0

    # Gaussian priors on tc1, k1
    sig_tc = 0.01
    sig_k  = 50.0
    lp += -0.5 * ((tc1 - TC_REF)**2 / sig_tc**2 + np.log(2*np.pi*sig_tc**2))
    lp += -0.5 * ((k1  - k1_init)**2 / sig_k**2  + np.log(2*np.pi*sig_k**2))

    # RV jitters priors
    lp += -0.5 * ((ln_jit_hires  - LN_JIT_CENTER)**2 / LN_JIT_SIGMA**2
                  + np.log(2*np.pi*LN_JIT_SIGMA**2))
    lp += -0.5 * ((ln_jit_harpsn - LN_JIT_CENTER)**2 / LN_JIT_SIGMA**2
                  + np.log(2*np.pi*LN_JIT_SIGMA**2))

    # soft priors on trend
    sig_g1 = 0.1
    sig_g2 = 1e-4
    lp += -0.5 * (g1**2 / sig_g1**2 + np.log(2*np.pi*sig_g1**2))
    lp += -0.5 * (g2**2 / sig_g2**2 + np.log(2*np.pi*sig_g2**2))

    # soft priors keeping e, ω near literature values
    sig_e = 0.02
    sig_w = np.deg2rad(5.0)
    lp += -0.5 * ((e_val - E_LIT)**2 / sig_e**2 + np.log(2*np.pi*sig_e**2))
    lp += -0.5 * ((omega_val - WRAD_LIT)**2 / sig_w**2 + np.log(2*np.pi*sig_w**2))

    return lp

def log_prob(theta, alpha_tess=1.0):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    lnL_rv   = rv_log_likelihood(theta)
    lnL_tess = tess_log_likelihood(theta)
    return lp + lnL_rv + alpha_tess * lnL_tess

def main() -> None:
    theta_start = theta_initial()
    ndim_joint  = len(theta_start)
    nwalkers    = 40

    scatter = np.array([
        1e-3,  # tc1
        5.0,   # k1
        1e-3,  # secosw1
        1e-3,  # sesinw1
        5.0,   # gamma_hires
        5.0,   # gamma_harpsn
        0.1,   # ln_jit_hires
        0.1,   # ln_jit_harpsn
        0.01,  # g1
        1e-5,  # g2
        1e-3,  # delta_t
        0.5,   # ln_jit_tess
        1e-3,  # rp
        1e-3,  # a
        0.1    # inc
    ])
    assert scatter.shape[0] == ndim_joint

    pos0 = theta_start + scatter * np.random.randn(nwalkers, ndim_joint)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim_joint, log_prob, kwargs={"alpha_tess": 0.5}
    )

    print("\nJOINT RV+TESS: burn-in...")
    sampler.run_mcmc(pos0, 4000, progress=True)
    sampler.reset()
    print("JOINT RV+TESS: production...")
    sampler.run_mcmc(None, 8000, progress=True)

    flat = sampler.get_chain(thin=10, flat=True)
    theta_med = np.percentile(flat, 50, axis=0)
    delta_t_med, ln_jit_tess_med = theta_to_params(theta_med)

    (tc1_med, k1_med, secosw_med, sesinw_med,
     gamma_hires_med, gamma_harpsn_med,
     ln_jit_hires_med, ln_jit_harpsn_med,
     g1_med, g2_med,
     delta_t_med, ln_jit_tess_med,
     rp_med, a_med, inc_med) = theta_med

    jit_hires_med  = np.exp(ln_jit_hires_med)
    jit_harpsn_med = np.exp(ln_jit_harpsn_med)
    jit_tess_med   = np.exp(ln_jit_tess_med)
    e_med, omega_med = e_omega_from_se(secosw_med, sesinw_med)

    # RV diagnostics
    rv_planet_med = mod_joint(t_rv)
    rv_model_med  = rv_planet_med.copy()
    rv_model_med[mask_hires]  += gamma_hires_med
    rv_model_med[mask_harpsn] += gamma_harpsn_med
    rv_model_med += g1_med * t_rel_rv + g2_med * (t_rel_rv**2)

    resid_rv = rv_all - rv_model_med
    sigma_hires_med  = np.sqrt(err_rv[mask_hires]**2  + jit_hires_med**2)
    sigma_harpsn_med = np.sqrt(err_rv[mask_harpsn]**2 + jit_harpsn_med**2)
    chi2_rv = (np.sum((resid_rv[mask_hires]  / sigma_hires_med)**2) +
               np.sum((resid_rv[mask_harpsn] / sigma_harpsn_med**2)))
    dof_rv   = len(t_rv) - ndim_joint
    chi2r_rv = chi2_rv / dof_rv

    # TESS diagnostics
    flux_model_med = tess_model(theta_med, t_win_bjd)
    sigma_tess_med = np.sqrt(ferr_win**2 + jit_tess_med**2)
    resid_tess = (f_win - flux_model_med) / sigma_tess_med
    chi2_tess = np.sum(resid_tess**2)
    dof_tess  = len(t_win_bjd) - 5   # tc1, delta_t, ln_jit_tess, rp, a, inc (approx)
    chi2r_tess = chi2_tess / dof_tess

    print("\nJOINT RV+TESS DIAGNOSTICS (shared orbit, epoch-linked)")
    print("=======================================================")
    print(f"tc1           = {tc1_med:.6f}")
    print(f"k1            = {k1_med:.3f} m/s")
    print(f"e             = {e_med:.4f}")
    print(f"omega (deg)   = {np.rad2deg(omega_med):.2f}")
    print(f"gamma_hires   = {gamma_hires_med:.3f} m/s")
    print(f"gamma_harpsn  = {gamma_harpsn_med:.3f} m/s")
    print(f"jit_hires     = {jit_hires_med:.3f} m/s")
    print(f"jit_harpsn    = {jit_harpsn_med:.3f} m/s")
    print(f"g1            = {g1_med:.6f} m/s/day")
    print(f"g2            = {g2_med:.9f} m/s/day^2")
    print(f"N_cycles      = {N_cycles}")
    print(f"delta_t       = {delta_t_med:.6f} days")
    print(f"jit_tess      = {jit_tess_med:.5f} (flux units)")
    print(f"rp            = {rp_med:.6f}")
    print(f"a (a/R*)      = {a_med:.6f}")
    print(f"inc           = {inc_med:.3f} deg")
    print(f"chi2_rv       = {chi2_rv:.2f}  (reduced ~ {chi2r_rv:.3f})")
    print(f"chi2_tess     = {chi2_tess:.2f} (reduced ~ {chi2r_tess:.3f})")

    # Optional plots
    phase_rv = phase_fold(t_rv, tc1_med, PERIOD)
    phase_all_rv = np.concatenate([phase_rv, phase_rv + 1.0])
    rv_all_2  = np.concatenate([rv_all, rv_all])
    err_rv_2  = np.concatenate([err_rv, err_rv])
    inst_2    = np.concatenate([inst_id, inst_id])

    mask_hires2  = (inst_2 == 0)
    mask_harpsn2 = (inst_2 == 1)

    phase_model = np.linspace(0.0, 2.0, 800)
    t_model_phase = tc1_med + phase_model * PERIOD
    rv_planet_curve = mod_joint(t_model_phase)
    rv_curve = rv_planet_curve + gamma_hires_med
    rv_curve += g1_med * (t_model_phase - time_base) + g2_med * (t_model_phase - time_base)**2

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].errorbar(phase_all_rv[mask_hires2], rv_all_2[mask_hires2],
                   yerr=err_rv_2[mask_hires2], fmt='o', ms=4, alpha=0.7, label='HIRES')
    ax[0].errorbar(phase_all_rv[mask_harpsn2], rv_all_2[mask_harpsn2],
                   yerr=err_rv_2[mask_harpsn2], fmt='s', ms=4, alpha=0.7, label='HARPS-N')
    ax[0].plot(phase_model, rv_curve, 'k-', lw=1.7, label='Joint median model')
    ax[0].set_ylabel('RV (m/s)')
    ax[0].set_title('HAT-P-2b: joint RV+TESS phase-folded RV')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    resid_rv_2 = np.concatenate([resid_rv, resid_rv])
    ax[1].errorbar(phase_all_rv[mask_hires2], resid_rv_2[mask_hires2],
                   yerr=err_rv_2[mask_hires2], fmt='o', ms=4, alpha=0.7)
    ax[1].errorbar(phase_all_rv[mask_harpsn2], resid_rv_2[mask_harpsn2],
                   yerr=err_rv_2[mask_harpsn2], fmt='s', ms=4, alpha=0.7)
    ax[1].axhline(0, color='r', ls='--', lw=1)
    ax[1].set_xlabel('Orbital phase')
    ax[1].set_ylabel('Residuals (m/s)')
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # data phases using the window already selected
    phase_win = ((t_win_bjd - t0_tess_bjd_est) / PERIOD) % 1.0
    phase_win = phase_win - np.round(phase_win)

    phase_window = 0.1
    mask_phase = np.abs(phase_win) < phase_window

    # model evaluated on a smooth phase grid
    phase_grid = np.linspace(-phase_window, phase_window, 1000)

    # ONE constant shift for the model curve
    phase_shift_model = 0.0045   # try 0.003 to 0.006 if needed

    # build model times from shifted phase grid
    t_grid_bjd = t0_tess_bjd_est + (phase_grid + phase_shift_model) * PERIOD
    flux_grid = tess_model(theta_med, t_grid_bjd)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))

    ax2.errorbar(phase_win[mask_phase], f_win[mask_phase],
                 yerr=ferr_win[mask_phase],
                 fmt=".k", ms=2, alpha=0.3,
                 label="TESS data (phase-folded, window)")

    ax2.plot(phase_grid, flux_grid, "r", lw=2, label="Joint model")

    ax2.set_xlabel("Orbital phase")
    ax2.set_ylabel("Normalized flux")
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()