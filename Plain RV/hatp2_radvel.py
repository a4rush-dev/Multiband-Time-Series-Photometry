import os

# NOTE: Python 3.13 sometimes has issues, use <3.10 when needed.
# Make RadVel work on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', os.path.expanduser('~'))

import numpy as np
import matplotlib.pyplot as plt
import radvel
from radvel import fitting
import radvel.likelihood as rlike
import warnings
warnings.filterwarnings('ignore')

import emcee

# =====================================================
# LOAD DATA: hat_p2_rv.txt
# BJD-2450000  RV(m/s)  e_RV(m/s)  inst (0=HIRES, 1=HARPS-N)
# =====================================================
data = np.loadtxt("hat_p2_rv.txt")
t_all    = data[:, 0] + 2450000.0
rv_all   = data[:, 1]
err_all  = data[:, 2]
inst_id  = data[:, 3].astype(int)

mask_hires  = (inst_id == 0)
mask_harpsn = (inst_id == 1)

time_base = np.median(t_all)
t_rel     = t_all - time_base   # for trends

# ORBITAL CONSTANTS (FROM LITERATURE)
per   = 5.6334729
tc    = 2454529.674
e0    = 0.502
wdeg0 = 188.8
wrad0 = np.deg2rad(wdeg0)

secosw0 = np.sqrt(e0) * np.cos(wrad0)
sesinw0 = np.sqrt(e0) * np.sin(wrad0)

# helper
def e_omega_from_se(secosw_val, sesinw_val):
    e_val = secosw_val**2 + sesinw_val**2
    omega_val = np.arctan2(sesinw_val, secosw_val)
    return e_val, omega_val

def phase_fold(t, tc_use, per_use):
    return np.mod((t - tc_use) / per_use, 1.0)

# jitter prior hyperparameters (common)
ln_jit_center = np.log(5.0)
ln_jit_sigma  = 0.7

# =====================================================
# MODEL 1: STATIC ONE-PLANET
# =====================================================
params_static = radvel.Parameters(1, basis="per tc secosw sesinw k")
params_static['per1']    = radvel.Parameter(value=per,    vary=False)
params_static['tc1']     = radvel.Parameter(value=tc,     vary=True)
params_static['secosw1'] = radvel.Parameter(value=secosw0, vary=False)
params_static['sesinw1'] = radvel.Parameter(value=sesinw0, vary=False)
params_static['k1']      = radvel.Parameter(value=950.0,  vary=True)

params_static['gamma_hires']  = radvel.Parameter(value=np.mean(rv_all[mask_hires]),
                                                 vary=True, linear=True)
params_static['jit_hires']    = radvel.Parameter(value=3.0, vary=True)
params_static['gamma_harpsn'] = radvel.Parameter(value=np.mean(rv_all[mask_harpsn]),
                                                 vary=True, linear=True)
params_static['jit_harpsn']   = radvel.Parameter(value=3.0, vary=True)

mod_static = radvel.RVModel(params_static)
mod_static.time_base = time_base

# RadVel maxlike for starting point
like_hires = rlike.RVLikelihood(
    mod_static, t_all[mask_hires], rv_all[mask_hires], err_all[mask_hires],
    suffix="_hires"
)
like_harpsn = rlike.RVLikelihood(
    mod_static, t_all[mask_harpsn], rv_all[mask_harpsn], err_all[mask_harpsn],
    suffix="_harpsn"
)

like_hires.params['gamma_hires']   = params_static['gamma_hires']
like_hires.params['jit_hires']     = params_static['jit_hires']
like_harpsn.params['gamma_harpsn'] = params_static['gamma_harpsn']
like_harpsn.params['jit_harpsn']   = params_static['jit_harpsn']

comp_like = rlike.CompositeLikelihood([like_hires, like_harpsn])
comp_like = fitting.maxlike_fitting(comp_like)
best_static = comp_like.params

print("\nSTATIC MODEL: MAXLIKE START")
for key in ['tc1', 'k1', 'gamma_hires', 'gamma_harpsn', 'jit_hires', 'jit_harpsn']:
    print(f"{key:14s} = {best_static[key].value:.6f}")

# theta_static = [tc1, k1, gamma_hires, gamma_harpsn, ln_jit_hires, ln_jit_harpsn]
def theta_from_params_static(pdict):
    return np.array([
        pdict['tc1'].value,
        pdict['k1'].value,
        pdict['gamma_hires'].value,
        pdict['gamma_harpsn'].value,
        np.log(pdict['jit_hires'].value),
        np.log(pdict['jit_harpsn'].value),
    ], dtype=float)

def update_params_from_theta_static(theta, pdict):
    tc1, k1, gamma_hires, gamma_harpsn, ln_jit_hires, ln_jit_harpsn = theta
    pdict['tc1'].value          = tc1
    pdict['k1'].value           = k1
    pdict['gamma_hires'].value  = gamma_hires
    pdict['gamma_harpsn'].value = gamma_harpsn
    pdict['jit_hires'].value    = np.exp(ln_jit_hires)
    pdict['jit_harpsn'].value   = np.exp(ln_jit_harpsn)

tc0   = tc
sigtc = 0.01
k0    = best_static['k1'].value
sigK  = 50.0

def log_prior_static(theta):
    tc1, k1, gamma_hires, gamma_harpsn, ln_jit_hires, ln_jit_harpsn = theta

    if not (tc0 - 0.5 < tc1 < tc0 + 0.5):
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

    lp  = -0.5 * ((tc1 - tc0)**2 / sigtc**2 + np.log(2.0 * np.pi * sigtc**2))
    lp += -0.5 * ((k1  - k0)**2  / sigK**2  + np.log(2.0 * np.pi * sigK**2))

    lp += -0.5 * ((ln_jit_hires  - ln_jit_center)**2 / ln_jit_sigma**2
                  + np.log(2.0 * np.pi * ln_jit_sigma**2))
    lp += -0.5 * ((ln_jit_harpsn - ln_jit_center)**2 / ln_jit_sigma**2
                  + np.log(2.0 * np.pi * ln_jit_sigma**2))
    return lp

def log_likelihood_static(theta):
    update_params_from_theta_static(theta, params_static)
    rv_planet = mod_static(t_all)

    tc1, k1, gamma_hires, gamma_harpsn, ln_jit_hires, ln_jit_harpsn = theta
    jit_hires  = np.exp(ln_jit_hires)
    jit_harpsn = np.exp(ln_jit_harpsn)

    rv_model = rv_planet.copy()
    rv_model[mask_hires]  += gamma_hires
    rv_model[mask_harpsn] += gamma_harpsn

    sigma_hires  = np.sqrt(err_all[mask_hires]**2  + jit_hires**2)
    sigma_harpsn = np.sqrt(err_all[mask_harpsn]**2 + jit_harpsn**2)

    resid_hires  = rv_all[mask_hires]  - rv_model[mask_hires]
    resid_harpsn = rv_all[mask_harpsn] - rv_model[mask_harpsn]

    lnL_hires  = -0.5 * np.sum((resid_hires / sigma_hires)**2
                               + np.log(2.0 * np.pi * sigma_hires**2))
    lnL_harpsn = -0.5 * np.sum((resid_harpsn / sigma_harpsn)**2
                               + np.log(2.0 * np.pi * sigma_harpsn**2))
    return lnL_hires + lnL_harpsn

def log_prob_static(theta):
    lp = log_prior_static(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_static(theta)

theta_best_static = theta_from_params_static(best_static)

ndim_static   = 6
nwalkers      = 32
scatter_vec   = np.array([1e-3, 5.0, 5.0, 5.0, 0.1, 0.1])
pos0_static   = theta_best_static + scatter_vec * np.random.randn(nwalkers, ndim_static)

sampler_static = emcee.EnsembleSampler(nwalkers, ndim_static, log_prob_static)

print("\nSTATIC MODEL: burn-in...")
sampler_static.run_mcmc(pos0_static, 4000, progress=True)
sampler_static.reset()
print("STATIC MODEL: production...")
sampler_static.run_mcmc(None, 8000, progress=True)

flat_static = sampler_static.get_chain(thin=10, flat=True)
theta_med_static = np.percentile(flat_static, 50, axis=0)
update_params_from_theta_static(theta_med_static, params_static)

rv_planet_med_static = mod_static(t_all)
tc_med_static        = params_static['tc1'].value
gamma_hires_med_static  = params_static['gamma_hires'].value
gamma_harpsn_med_static = params_static['gamma_harpsn'].value
jit_hires_med_static    = params_static['jit_hires'].value
jit_harpsn_med_static   = params_static['jit_harpsn'].value

rv_model_med_static = rv_planet_med_static.copy()
rv_model_med_static[mask_hires]  += gamma_hires_med_static
rv_model_med_static[mask_harpsn] += gamma_harpsn_med_static

resid_static = rv_all - rv_model_med_static
sigma_hires_static  = np.sqrt(err_all[mask_hires]**2  + jit_hires_med_static**2)
sigma_harpsn_static = np.sqrt(err_all[mask_harpsn]**2 + jit_harpsn_med_static**2)
chi2_static = (np.sum((resid_static[mask_hires]  / sigma_hires_static)**2) +
               np.sum((resid_static[mask_harpsn] / sigma_harpsn_static)**2))
dof_static  = len(t_all) - ndim_static
chi2r_static = chi2_static / dof_static

print("\nSTATIC MODEL DIAGNOSTICS")
print("=========================")
print(f"tc1           = {tc_med_static:.6f}")
print(f"k1            = {params_static['k1'].value:.3f} m/s")
print(f"gamma_hires   = {gamma_hires_med_static:.3f} m/s")
print(f"gamma_harpsn  = {gamma_harpsn_med_static:.3f} m/s")
print(f"jit_hires     = {jit_hires_med_static:.3f} m/s")
print(f"jit_harpsn    = {jit_harpsn_med_static:.3f} m/s")
print(f"chi2_med      = {chi2_static:.2f}")
print(f"chi2_r_med    = {chi2r_static:.3f}")
print(f"RMS_all       = {np.std(resid_static):.2f} m/s")

# =====================================================
# MODEL 2: "EVOLVING" (geometry varies with time)
# =====================================================
params_evol = radvel.Parameters(1, basis="per tc secosw sesinw k")
params_evol['per1']    = radvel.Parameter(value=per,    vary=False)
params_evol['tc1']     = radvel.Parameter(value=tc_med_static, vary=True)
params_evol['secosw1'] = radvel.Parameter(value=secosw0, vary=True)
params_evol['sesinw1'] = radvel.Parameter(value=sesinw0, vary=True)
params_evol['k1']      = radvel.Parameter(value=params_static['k1'].value, vary=True)

params_evol['gamma_hires']  = radvel.Parameter(value=gamma_hires_med_static,
                                               vary=True, linear=True)
params_evol['jit_hires']    = radvel.Parameter(value=jit_hires_med_static, vary=True)
params_evol['gamma_harpsn'] = radvel.Parameter(value=gamma_harpsn_med_static,
                                               vary=True, linear=True)
params_evol['jit_harpsn']   = radvel.Parameter(value=jit_harpsn_med_static, vary=True)

mod_evol = radvel.RVModel(params_evol)
mod_evol.time_base = time_base

def theta_from_params_evol(pdict):
    return np.array([
        pdict['tc1'].value,
        pdict['k1'].value,
        pdict['secosw1'].value,
        pdict['sesinw1'].value,
        pdict['gamma_hires'].value,
        pdict['gamma_harpsn'].value,
        np.log(pdict['jit_hires'].value),
        np.log(pdict['jit_harpsn'].value),
    ], dtype=float)

def update_params_from_theta_evol(theta, pdict):
    (tc1, k1, secosw1, sesinw1,
     gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn) = theta
    pdict['tc1'].value          = tc1
    pdict['k1'].value           = k1
    pdict['secosw1'].value      = secosw1
    pdict['sesinw1'].value      = sesinw1
    pdict['gamma_hires'].value  = gamma_hires
    pdict['gamma_harpsn'].value = gamma_harpsn
    pdict['jit_hires'].value    = np.exp(ln_jit_hires)
    pdict['jit_harpsn'].value   = np.exp(ln_jit_harpsn)

def log_prior_evol(theta):
    (tc1, k1, secosw1, sesinw1,
     gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn) = theta

    if not (tc0 - 0.5 < tc1 < tc0 + 0.5):
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

    # e in (0,1); secosw/sesinw close to literature values
    e_val, omega_val = e_omega_from_se(secosw1, sesinw1)
    if not (0.0 < e_val < 0.999):
        return -np.inf
    # very tight priors to prevent pathological e, ω
    sige = 0.02
    sigw = np.deg2rad(5.0)
    lp  = -0.5 * ((tc1 - tc0)**2 / sigtc**2 + np.log(2.0 * np.pi * sigtc**2))
    lp += -0.5 * ((k1  - k0)**2  / sigK**2  + np.log(2.0 * np.pi * sigK**2))
    lp += -0.5 * ((e_val - e0)**2 / sige**2 + np.log(2.0 * np.pi * sige**2))
    lp += -0.5 * ((omega_val - wrad0)**2 / sigw**2 + np.log(2.0 * np.pi * sigw**2))

    lp += -0.5 * ((ln_jit_hires  - ln_jit_center)**2 / ln_jit_sigma**2
                  + np.log(2.0 * np.pi * ln_jit_sigma**2))
    lp += -0.5 * ((ln_jit_harpsn - ln_jit_center)**2 / ln_jit_sigma**2
                  + np.log(2.0 * np.pi * ln_jit_sigma**2))
    return lp

def log_likelihood_evol(theta):
    update_params_from_theta_evol(theta, params_evol)
    rv_planet = mod_evol(t_all)

    (tc1, k1, secosw1, sesinw1,
     gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn) = theta
    jit_hires  = np.exp(ln_jit_hires)
    jit_harpsn = np.exp(ln_jit_harpsn)

    rv_model = rv_planet.copy()
    rv_model[mask_hires]  += gamma_hires
    rv_model[mask_harpsn] += gamma_harpsn

    sigma_hires  = np.sqrt(err_all[mask_hires]**2  + jit_hires**2)
    sigma_harpsn = np.sqrt(err_all[mask_harpsn]**2 + jit_harpsn**2)

    resid_hires  = rv_all[mask_hires]  - rv_model[mask_hires]
    resid_harpsn = rv_all[mask_harpsn] - rv_model[mask_harpsn]

    lnL_hires  = -0.5 * np.sum((resid_hires / sigma_hires)**2
                               + np.log(2.0 * np.pi * sigma_hires**2))
    lnL_harpsn = -0.5 * np.sum((resid_harpsn / sigma_harpsn)**2
                               + np.log(2.0 * np.pi * sigma_harpsn**2))
    return lnL_hires + lnL_harpsn

def log_prob_evol(theta):
    lp = log_prior_evol(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_evol(theta)

theta_start_evol = theta_from_params_evol(params_evol)
ndim_evol        = 8
pos0_evol        = theta_start_evol + np.array([1e-3, 5.0, 1e-3, 1e-3, 5.0, 5.0, 0.1, 0.1]) \
                   * np.random.randn(nwalkers, ndim_evol)

sampler_evol = emcee.EnsembleSampler(nwalkers, ndim_evol, log_prob_evol)

print("\nEVOLVING MODEL: burn-in...")
sampler_evol.run_mcmc(pos0_evol, 4000, progress=True)
sampler_evol.reset()
print("EVOLVING MODEL: production...")
sampler_evol.run_mcmc(None, 8000, progress=True)

flat_evol = sampler_evol.get_chain(thin=10, flat=True)
theta_med_evol = np.percentile(flat_evol, 50, axis=0)
update_params_from_theta_evol(theta_med_evol, params_evol)

rv_planet_med_evol = mod_evol(t_all)
tc_med_evol        = params_evol['tc1'].value
gamma_hires_med_evol  = params_evol['gamma_hires'].value
gamma_harpsn_med_evol = params_evol['gamma_harpsn'].value
jit_hires_med_evol    = params_evol['jit_hires'].value
jit_harpsn_med_evol   = params_evol['jit_harpsn'].value

rv_model_med_evol = rv_planet_med_evol.copy()
rv_model_med_evol[mask_hires]  += gamma_hires_med_evol
rv_model_med_evol[mask_harpsn] += gamma_harpsn_med_evol

resid_evol = rv_all - rv_model_med_evol
sigma_hires_evol  = np.sqrt(err_all[mask_hires]**2  + jit_hires_med_evol**2)
sigma_harpsn_evol = np.sqrt(err_all[mask_harpsn]**2 + jit_harpsn_med_evol**2)
chi2_evol = (np.sum((resid_evol[mask_hires]  / sigma_hires_evol)**2) +
             np.sum((resid_evol[mask_harpsn] / sigma_harpsn_evol)**2))
dof_evol   = len(t_all) - ndim_evol
chi2r_evol = chi2_evol / dof_evol

e_med, omega_med = e_omega_from_se(params_evol['secosw1'].value,
                                   params_evol['sesinw1'].value)

print("\nEVOLVING MODEL DIAGNOSTICS")
print("===========================")
print(f"tc1           = {tc_med_evol:.6f}")
print(f"k1            = {params_evol['k1'].value:.3f} m/s")
print(f"e             = {e_med:.4f}")
print(f"omega (deg)   = {np.rad2deg(omega_med):.2f}")
print(f"gamma_hires   = {gamma_hires_med_evol:.3f} m/s")
print(f"gamma_harpsn  = {gamma_harpsn_med_evol:.3f} m/s")
print(f"jit_hires     = {jit_hires_med_evol:.3f} m/s")
print(f"jit_harpsn    = {jit_harpsn_med_evol:.3f} m/s")
print(f"chi2_med      = {chi2_evol:.2f}")
print(f"chi2_r_med    = {chi2r_evol:.3f}")
print(f"RMS_all       = {np.std(resid_evol):.2f} m/s")

# =====================================================
# MODEL 3: QUADRATIC-TREND ONE-PLANET
# =====================================================
params_quad = radvel.Parameters(1, basis="per tc secosw sesinw k")
params_quad['per1']    = radvel.Parameter(value=per,    vary=False)
params_quad['tc1']     = radvel.Parameter(value=tc_med_static, vary=True)
params_quad['secosw1'] = radvel.Parameter(value=secosw0, vary=False)
params_quad['sesinw1'] = radvel.Parameter(value=sesinw0, vary=False)
params_quad['k1']      = radvel.Parameter(value=params_static['k1'].value, vary=True)

params_quad['gamma_hires']  = radvel.Parameter(value=gamma_hires_med_static,
                                               vary=True, linear=True)
params_quad['jit_hires']    = radvel.Parameter(value=jit_hires_med_static, vary=True)
params_quad['gamma_harpsn'] = radvel.Parameter(value=gamma_harpsn_med_static,
                                               vary=True, linear=True)
params_quad['jit_harpsn']   = radvel.Parameter(value=jit_harpsn_med_static, vary=True)

# global linear + quadratic trend
params_quad['g1'] = radvel.Parameter(value=0.0, vary=True, linear=True)
params_quad['g2'] = radvel.Parameter(value=0.0, vary=True, linear=True)

mod_quad = radvel.RVModel(params_quad)
mod_quad.time_base = time_base

def theta_from_params_quad(pdict):
    return np.array([
        pdict['tc1'].value,
        pdict['k1'].value,
        pdict['gamma_hires'].value,
        pdict['gamma_harpsn'].value,
        np.log(pdict['jit_hires'].value),
        np.log(pdict['jit_harpsn'].value),
        pdict['g1'].value,
        pdict['g2'].value,
    ], dtype=float)

def update_params_from_theta_quad(theta, pdict):
    (tc1, k1, gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn,
     g1, g2) = theta
    pdict['tc1'].value          = tc1
    pdict['k1'].value           = k1
    pdict['gamma_hires'].value  = gamma_hires
    pdict['gamma_harpsn'].value = gamma_harpsn
    pdict['jit_hires'].value    = np.exp(ln_jit_hires)
    pdict['jit_harpsn'].value   = np.exp(ln_jit_harpsn)
    pdict['g1'].value           = g1
    pdict['g2'].value           = g2

def log_prior_quad(theta):
    (tc1, k1, gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn,
     g1, g2) = theta

    if not (tc0 - 0.5 < tc1 < tc0 + 0.5):
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

    # conservative bounds on trend coefficients
    if not (-1.0 < g1 < 1.0):
        return -np.inf
    if not (-1e-3 < g2 < 1e-3):
        return -np.inf

    lp  = -0.5 * ((tc1 - tc0)**2 / sigtc**2 + np.log(2.0 * np.pi * sigtc**2))
    lp += -0.5 * ((k1  - k0)**2  / sigK**2  + np.log(2.0 * np.pi * sigK**2))

    lp += -0.5 * ((ln_jit_hires  - ln_jit_center)**2 / ln_jit_sigma**2
                  + np.log(2.0 * np.pi * ln_jit_sigma**2))
    lp += -0.5 * ((ln_jit_harpsn - ln_jit_center)**2 / ln_jit_sigma**2
                  + np.log(2.0 * np.pi * ln_jit_sigma**2))

    sig_g1 = 0.1
    sig_g2 = 1e-4
    lp += -0.5 * (g1**2 / sig_g1**2 + np.log(2.0 * np.pi * sig_g1**2))
    lp += -0.5 * (g2**2 / sig_g2**2 + np.log(2.0 * np.pi * sig_g2**2))
    return lp

def log_likelihood_quad(theta):
    update_params_from_theta_quad(theta, params_quad)
    rv_planet = mod_quad(t_all)

    (tc1, k1, gamma_hires, gamma_harpsn,
     ln_jit_hires, ln_jit_harpsn,
     g1, g2) = theta

    jit_hires  = np.exp(ln_jit_hires)
    jit_harpsn = np.exp(ln_jit_harpsn)

    rv_model = rv_planet.copy()
    rv_model[mask_hires]  += gamma_hires
    rv_model[mask_harpsn] += gamma_harpsn

    trend = g1 * t_rel + g2 * (t_rel**2)
    rv_model += trend

    sigma_hires  = np.sqrt(err_all[mask_hires]**2  + jit_hires**2)
    sigma_harpsn = np.sqrt(err_all[mask_harpsn]**2 + jit_harpsn**2)

    resid_hires  = rv_all[mask_hires]  - rv_model[mask_hires]
    resid_harpsn = rv_all[mask_harpsn] - rv_model[mask_harpsn]

    lnL_hires  = -0.5 * np.sum((resid_hires / sigma_hires)**2
                               + np.log(2.0 * np.pi * sigma_hires**2))
    lnL_harpsn = -0.5 * np.sum((resid_harpsn / sigma_harpsn)**2
                               + np.log(2.0 * np.pi * sigma_harpsn**2))
    return lnL_hires + lnL_harpsn

def log_prob_quad(theta):
    lp = log_prior_quad(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_quad(theta)

theta_start_quad = theta_from_params_quad(params_quad)
ndim_quad        = 8
pos0_quad        = theta_start_quad + np.array([1e-3, 5.0, 5.0, 5.0, 0.1, 0.1, 0.01, 1e-5]) \
                   * np.random.randn(nwalkers, ndim_quad)

sampler_quad = emcee.EnsembleSampler(nwalkers, ndim_quad, log_prob_quad)

print("\nQUADRATIC MODEL: burn-in...")
sampler_quad.run_mcmc(pos0_quad, 4000, progress=True)
sampler_quad.reset()
print("QUADRATIC MODEL: production...")
sampler_quad.run_mcmc(None, 8000, progress=True)

flat_quad = sampler_quad.get_chain(thin=10, flat=True)
theta_med_quad = np.percentile(flat_quad, 50, axis=0)
update_params_from_theta_quad(theta_med_quad, params_quad)

rv_planet_med_quad = mod_quad(t_all)
tc_med_quad        = params_quad['tc1'].value
gamma_hires_med_quad  = params_quad['gamma_hires'].value
gamma_harpsn_med_quad = params_quad['gamma_harpsn'].value
jit_hires_med_quad    = params_quad['jit_hires'].value
jit_harpsn_med_quad   = params_quad['jit_harpsn'].value
g1_med               = params_quad['g1'].value
g2_med               = params_quad['g2'].value

rv_model_med_quad = rv_planet_med_quad.copy()
rv_model_med_quad[mask_hires]  += gamma_hires_med_quad
rv_model_med_quad[mask_harpsn] += gamma_harpsn_med_quad
rv_model_med_quad += g1_med * t_rel + g2_med * (t_rel**2)

resid_quad = rv_all - rv_model_med_quad
sigma_hires_quad  = np.sqrt(err_all[mask_hires]**2  + jit_hires_med_quad**2)
sigma_harpsn_quad = np.sqrt(err_all[mask_harpsn]**2 + jit_harpsn_med_quad**2)
chi2_quad = (np.sum((resid_quad[mask_hires]  / sigma_hires_quad)**2) +
             np.sum((resid_quad[mask_harpsn] / sigma_harpsn_quad)**2))
dof_quad   = len(t_all) - ndim_quad
chi2r_quad = chi2_quad / dof_quad

print("\nQUADRATIC MODEL DIAGNOSTICS")
print("============================")
print(f"tc1           = {tc_med_quad:.6f}")
print(f"k1            = {params_quad['k1'].value:.3f} m/s")
print(f"gamma_hires   = {gamma_hires_med_quad:.3f} m/s")
print(f"gamma_harpsn  = {gamma_harpsn_med_quad:.3f} m/s")
print(f"jit_hires     = {jit_hires_med_quad:.3f} m/s")
print(f"jit_harpsn    = {jit_harpsn_med_quad:.3f} m/s")
print(f"g1            = {g1_med:.6f} m/s/day")
print(f"g2            = {g2_med:.9f} m/s/day^2")
print(f"chi2_med      = {chi2_quad:.2f}")
print(f"chi2_r_med    = {chi2r_quad:.3f}")
print(f"RMS_all       = {np.std(resid_quad):.2f} m/s")

# =====================================================
# PHASE-FOLDED PLOTS FOR ALL THREE MODELS
# =====================================================
def make_phase_plot(name, tc_med, rv_model_med, resid_med, tc_curve_med, mod_obj,
                    gamma_for_curve, extra_trend=None, filename="out.png"):
    phase = phase_fold(t_all, tc_med, per)
    phase_all = np.concatenate([phase, phase + 1.0])
    rv_all_2  = np.concatenate([rv_all, rv_all])
    err_all_2 = np.concatenate([err_all, err_all])
    inst_2    = np.concatenate([inst_id, inst_id])
    resid_2   = np.concatenate([resid_med, resid_med])

    phase_model = np.linspace(0.0, 2.0, 800)
    t_model_phase = tc_curve_med + phase_model * per
    rv_planet_curve = mod_obj(t_model_phase)
    rv_curve = rv_planet_curve + gamma_for_curve
    if extra_trend is not None:
        rv_curve += extra_trend(t_model_phase)

    mask_hires2  = (inst_2 == 0)
    mask_harpsn2 = (inst_2 == 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].errorbar(phase_all[mask_hires2], rv_all_2[mask_hires2],
                   yerr=err_all_2[mask_hires2],
                   fmt='o', ms=4, alpha=0.7, label='HIRES')
    ax[0].errorbar(phase_all[mask_harpsn2], rv_all_2[mask_harpsn2],
                   yerr=err_all_2[mask_harpsn2],
                   fmt='s', ms=4, alpha=0.7, label='HARPS-N')
    ax[0].plot(phase_model, rv_curve, 'k-', lw=1.7, label=name + " median model")
    ax[0].set_ylabel('RV (m/s)')
    ax[0].set_title(f'HAT-P-2b: phase-folded RV ({name})')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].errorbar(phase_all[mask_hires2], resid_2[mask_hires2],
                   yerr=err_all_2[mask_hires2],
                   fmt='o', ms=4, alpha=0.7)
    ax[1].errorbar(phase_all[mask_harpsn2], resid_2[mask_harpsn2],
                   yerr=err_all_2[mask_harpsn2],
                   fmt='s', ms=4, alpha=0.7)
    ax[1].axhline(0, color='r', ls='--', lw=1)
    ax[1].set_xlabel('Orbital phase')
    ax[1].set_ylabel('Residuals (m/s)')
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)

# static
make_phase_plot(
    name="static one-planet",
    tc_med=tc_med_static,
    rv_model_med=rv_model_med_static,
    resid_med=resid_static,
    tc_curve_med=tc_med_static,
    mod_obj=mod_static,
    gamma_for_curve=gamma_hires_med_static,
    extra_trend=None,
    filename="hatp2_rv_static_phase.png",
)

# evolving
make_phase_plot(
    name="e/ω-free one-planet",
    tc_med=tc_med_evol,
    rv_model_med=rv_model_med_evol,
    resid_med=resid_evol,
    tc_curve_med=tc_med_evol,
    mod_obj=mod_evol,
    gamma_for_curve=gamma_hires_med_evol,
    extra_trend=None,
    filename="hatp2_rv_evolving_phase.png",
)

# quadratic
def quad_trend_func(tarr):
    return g1_med * (tarr - time_base) + g2_med * (tarr - time_base)**2

make_phase_plot(
    name="one-planet + quadratic trend",
    tc_med=tc_med_quad,
    rv_model_med=rv_model_med_quad,
    resid_med=resid_quad,
    tc_curve_med=tc_med_quad,
    mod_obj=mod_quad,
    gamma_for_curve=gamma_hires_med_quad,
    extra_trend=quad_trend_func,
    filename="hatp2_rv_quadratic_phase.png",
)

print("\nDone: static, e/ω-free, and quadratic one-planet phase plots saved.")