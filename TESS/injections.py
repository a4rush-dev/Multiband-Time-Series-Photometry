import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import batman
from astropy.timeseries import LombScargle

# ========== LOAD DATA ==========
df = pd.read_csv("hatp2_tess_lightcurve.csv")
t = np.array(df["time"])
f = np.array(df["flux"])
ferr = np.array(df["flux_err"])

if np.any(~np.isfinite(ferr)) or np.any(ferr <= 0):
    ferr = np.ones_like(f) * np.std(f) * 1.0

median_flux = np.median(f)
f = f / median_flux
ferr = ferr / median_flux

period = 5.6334729
orbital_freq = 1.0 / period

# ========== FIND T0 BY GRID SEARCH ==========
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
print(f"Estimated t0 = {t0_est:.6f}")

with open("tess_params.json", "r") as f_json:
    params_dict = json.load(f_json)

t0_est = params_dict["t0_est"]
t0_rel = params_dict["t0_rel"]
rp = params_dict["rp"]
a = params_dict["a"]
inc = params_dict["inc"]

print(f"Parameters loaded from tess_params.json")
print(f"  t0_est: {t0_est:.6f}")
print(f"  t0_rel: {t0_rel:.6f}")
print(f"  rp: {rp:.6f}")
print(f"  a: {a:.6f}")
print(f"  inc: {inc:.6f}\n")

params = batman.TransitParams()
params.t0 = 0.0
params.per = period
params.rp = rp
params.a = a
params.inc = inc
params.ecc = 0.516
params.w = 188.0
params.limb_dark = "quadratic"
params.u = [0.2, 0.3]

def model(theta, t_rel):
    t0_rel_val, rp_val, a_val, inc_val = theta
    params.t0 = t0_rel_val
    params.rp = rp_val
    params.a = a_val
    params.inc = inc_val
    m = batman.TransitModel(params, t_rel, supersample_factor=5, exp_time=(t_rel[1]-t_rel[0] if len(t_rel)>1 else 0.002))
    return m.light_curve(params)

best = np.array([t0_rel, rp, a, inc])
best_flux = model(best, t - t0_est)

# ========== INJECTION ==========
f_79 = 79 * orbital_freq
f_91 = 91 * orbital_freq

A_79, A_91 = 30e-6, 30e-6  # for demo, use as needed

phi_79 = np.random.uniform(0, 2*np.pi)
phi_91 = np.random.uniform(0, 2*np.pi)
signal_79 = A_79 * np.cos(2*np.pi * f_79 * t + phi_79)
signal_91 = A_91 * np.cos(2*np.pi * f_91 * t + phi_91)
flux_inj = f + signal_79 + signal_91

# ========== PERIODOGRAMS (UNFOLDED) ==========
residuals = f - best_flux + 1.0
residuals_inj = flux_inj - best_flux + 1.0

probabilities = [0.0455, 0.0027, 6.334e-5, 5.733e-7]

ls = LombScargle(t, residuals)
freq, power = ls.autopower(minimum_frequency=0.01, maximum_frequency=20, samples_per_peak=10)
alarm = ls.false_alarm_level(probabilities)

ls_inj = LombScargle(t, residuals_inj)
freq_inj, power_inj = ls_inj.autopower(minimum_frequency=0.01, maximum_frequency=20, samples_per_peak=10)
alarm_inj = ls_inj.false_alarm_level(probabilities)

# ========== FOLDED  ==========
t_fold = t % period
sort_fold = np.argsort(t_fold)
t_fold_sorted = t_fold[sort_fold]

flux_fold = f[sort_fold]
flux_inj_fold = flux_inj[sort_fold]
ferr_fold = ferr[sort_fold]

# BUILD MODEL ON UNFOLDED TIME, THEN SORT IT
model_full = model(best, t - t0_est)  # Model evaluated on full unfolded times
model_fold = model_full[sort_fold]     # Reorder by the fold-sort indices

# Residuals on folded times
residuals_fold = flux_fold - model_fold + 1.0
residuals_inj_fold = flux_inj_fold - model_fold + 1.0

ls_fold = LombScargle(t_fold_sorted, residuals_fold, ferr_fold)
freq_fold, power_fold = ls_fold.autopower(minimum_frequency=0.01, maximum_frequency=20, samples_per_peak=10)
alarm_fold = ls_fold.false_alarm_level(probabilities)

ls_inj_fold = LombScargle(t_fold_sorted, residuals_inj_fold, ferr_fold)
freq_inj_fold, power_inj_fold = ls_inj_fold.autopower(minimum_frequency=0.01, maximum_frequency=20, samples_per_peak=10)
alarm_inj_fold = ls_inj_fold.false_alarm_level(probabilities)

# ========== PLOTS ==========

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(t_fold_sorted, flux_fold, ".", ms=2, alpha=0.4, label='Folded TESS flux')
ax1.plot(t_fold_sorted, model_fold, 'r-', lw=2, label='Best-fit Batman model')
ax1.set_xlabel("Folded Time (days)")
ax1.set_ylabel("Normalized Flux")
ax1.set_title("Folded TESS Light Curve (No Injection)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(t_fold_sorted, flux_inj_fold, ".", ms=2, alpha=0.4, label='Folded Injected Flux')
ax2.plot(t_fold_sorted, model_fold, 'r-', lw=2, label='Best-fit Batman model')
ax2.set_xlabel("Folded Time (days)")
ax2.set_ylabel("Normalized Flux")
ax2.set_title("Folded TESS Light Curve (Injected)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
labels = ['~10% FAP', '~1% FAP', '~0.01% FAP', '~0.0001% FAP']
colors = ['red', 'orange', 'purple', 'darkred']

# Top-Left: Unfolded, no injection
ax1.plot(freq, power, 'k-', lw=1.5, alpha=0.7, label='LS periodogram')
for amp, label_, color in zip(alarm, labels, colors):
    ax1.axhline(amp, color=color, linestyle='--', alpha=0.5, label=label_)
ax1.axvline(f_79, color='blue', linestyle='--', lw=2, label=f'79f ({f_79:.2f} d$^{{-1}}$)')
ax1.axvline(f_91, color='green', linestyle='--', lw=2, label=f'91f ({f_91:.2f} d$^{{-1}}$)')
ax1.set_ylabel('Lomb-Scargle Power', fontsize=12)
ax1.set_title('UNFOLDED - No Injections', fontsize=12)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 20)

# Top-Right: Unfolded, with injection
ax2.plot(freq_inj, power_inj, 'k-', lw=1.5, alpha=0.7, label='LS periodogram')
for amp, label_, color in zip(alarm_inj, labels, colors):
    ax2.axhline(amp, color=color, linestyle='--', alpha=0.5, label=label_)
ax2.axvline(f_79, color='blue', linestyle='--', lw=2, label=f'79f ({f_79:.2f} d$^{{-1}}$)')
ax2.axvline(f_91, color='green', linestyle='--', lw=2, label=f'91f ({f_91:.2f} d$^{{-1}}$)')
ax2.set_ylabel('Lomb-Scargle Power', fontsize=12)
ax2.set_title('UNFOLDED - With Injection', fontsize=12)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 20)

# Bottom-Left: Folded, no injection
ax3.plot(freq_fold, power_fold, 'k-', lw=1.5, alpha=0.7, label='LS periodogram')
for amp, label_, color in zip(alarm_fold, labels, colors):
    ax3.axhline(amp, color=color, linestyle='--', alpha=0.5, label=label_)
ax3.axvline(f_79, color='blue', linestyle='--', lw=2, label=f'79f ({f_79:.2f} d$^{{-1}}$)')
ax3.axvline(f_91, color='green', linestyle='--', lw=2, label=f'91f ({f_91:.2f} d$^{{-1}}$)')
ax3.set_xlabel('Frequency (d$^{-1}$)', fontsize=12)
ax3.set_ylabel('Lomb-Scargle Power', fontsize=12)
ax3.set_title('FOLDED - No Injections', fontsize=12)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 20)

# Bottom-Right: Folded, with injection
ax4.plot(freq_inj_fold, power_inj_fold, 'k-', lw=1.5, alpha=0.7, label='LS periodogram')
for amp, label_, color in zip(alarm_inj_fold, labels, colors):
    ax4.axhline(amp, color=color, linestyle='--', alpha=0.5, label=label_)
ax4.axvline(f_79, color='blue', linestyle='--', lw=2, label=f'79f ({f_79:.2f} d$^{{-1}}$)')
ax4.axvline(f_91, color='green', linestyle='--', lw=2, label=f'91f ({f_91:.2f} d$^{{-1}}$)')
ax4.set_xlabel('Frequency (d$^{-1}$)', fontsize=12)
ax4.set_ylabel('Lomb-Scargle Power', fontsize=12)
ax4.set_title('FOLDED - With Injection', fontsize=12)
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 20)

plt.suptitle('', fontsize=12, y=0.995)
plt.tight_layout()
plt.show()