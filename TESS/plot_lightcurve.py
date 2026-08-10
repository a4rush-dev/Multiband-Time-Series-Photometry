import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batman
import emcee
from scipy import signal

# load
df = pd.read_csv("hatp2_tess_lightcurve.csv")
t = np.array(df["time"])        # in days
f = np.array(df["flux"])
ferr = np.array(df["flux_err"])

# quick sanity: replace zero or nan errors with scatter of out-of-transit
if np.any(~np.isfinite(ferr)) or np.any(ferr <= 0):
    ferr = np.ones_like(f) * np.std(f) * 1.0
    print("warning: flux_err had bad values -> replaced with global scatter estimate")

# normalize flux
median_flux = np.median(f)
f = f / median_flux
ferr = ferr / median_flux

# known period (HAT-P-2b)
period = 5.6334729

# === quick grid search to estimate t0 ===
ntrial = 1200
trial_offsets = np.linspace(0, period, ntrial)
window_phase = 0.02
depths = np.zeros(ntrial)

for i, off in enumerate(trial_offsets):
    phase = ((t - off) / period) % 1.0
    phase = phase - np.round(phase)
    mask = np.abs(phase) < window_phase
    if np.sum(mask) < 5:
        depths[i] = np.nan
    else:
        depths[i] = np.nanmedian(f[mask])

best_idx = np.nanargmin(depths)
t0_est = trial_offsets[best_idx]
print("estimated t0 (offset mod P) =", t0_est)

# window around transit
phase = ((t - t0_est) / period) % 1.0
phase = phase - np.round(phase)
window = 0.05
mask_window = np.abs(phase) < window

t_win = (t[mask_window] - t0_est)
f_win = f[mask_window]
ferr_win = ferr[mask_window]

order = np.argsort(t_win)
t_win = t_win[order]
f_win = f_win[order]
ferr_win = ferr_win[order]

# === batman setup ===
params = batman.TransitParams()
params.t0 = 0.0
params.per = period
params.rp = 0.0722
params.a = 8.9
params.inc = 86.3
params.ecc = 0.516
params.w = 188.0
params.limb_dark = "quadratic"
params.u = [0.2, 0.3]

def model(theta, t_rel):
    t0_rel, rp, a, inc = theta
    params.t0 = t0_rel
    params.rp = rp
    params.a = a
    params.inc = inc
    m = batman.TransitModel(params, t_rel, supersample_factor=5,
                            exp_time=(t_rel[1]-t_rel[0] if len(t_rel)>1 else 0.002))
    return m.light_curve(params)

def log_likelihood(theta, t_rel, f_rel, ferr_rel):
    model_flux = model(theta, t_rel)
    res = (f_rel - model_flux) / ferr_rel
    return -0.5 * np.sum(res**2 + np.log(2*np.pi*ferr_rel**2))

def log_prior(theta):
    t0_rel, rp, a, inc = theta
    if -0.5 < t0_rel < 0.5 and 0.04 < rp < 0.11 and 5.0 < a < 15.0 and 80.0 < inc < 90.0:
        return 0.0
    return -np.inf

def log_prob(theta, t_rel, f_rel, ferr_rel):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t_rel, f_rel, ferr_rel)

# === run emcee ===
ndim = 4
nwalkers = 24
init = np.array([0.0, 0.0722, 8.9, 86.3])
pos = init + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(t_win, f_win, ferr_win))
print("running short test (1000 steps)...")
sampler.run_mcmc(pos, 1000, progress=True)

flat = sampler.get_chain(discard=200, thin=5, flat=True)

# === best fit ===
best = np.median(flat, axis=0)
t0_rel, rp, a, inc = best

# PRINT THE BEST FIT PARAMETERS
print("\n" + "="*50)
print("BEST FIT PARAMETERS FROM TESS")
print("="*50)
print(f"t0_rel (relative) = {t0_rel:.6f}")
print(f"t0_absolute = {t0_est + t0_rel:.6f}")
print(f"rp = {rp:.6f}")
print(f"a = {a:.6f}")
print(f"inc = {inc:.6f}")
print(f"period = {period} (fixed)")
print(f"ecc = {params.ecc} (fixed)")
print(f"w = {params.w} (fixed)")
print(f"u = {params.u} (fixed)")
print("="*50)

best_flux = model(best, t - t0_est)

# Display the plot
plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(10,5))
ax1.errorbar(t, f, yerr=ferr, fmt=".k", ms=2, alpha=0.3, label="TESS data")
ax1.plot(t, best_flux, "r", lw=2, label="best-fit model")
ax1.set_ylabel("Normalized flux")
ax1.invert_yaxis()
ax1.legend()
plt.tight_layout()
plt.show()