import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batman
from astropy.timeseries import LombScargle
import json
import os

# load data
df = pd.read_csv("hatp2_tess_lightcurve.csv")
t = np.array(df["time"])
f = np.array(df["flux"])
ferr = np.array(df["flux_err"])

if np.any(~np.isfinite(ferr)) or np.any(ferr <= 0):
    ferr = np.ones_like(f) * np.std(f) * 1.0
    print("warning: flux_err had bad values -> replaced with global scatter estimate")

median_flux = np.median(f)
f = f / median_flux
ferr = ferr / median_flux

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
t0_est = float(trial_offsets[best_idx])  # Make sure it's a float (for later json type-handling)
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

# ========== LOAD PARAMETERS FROM JSON ==========
if not os.path.exists("tess_params.json"):
    raise RuntimeError("tess_params.json not found! Run your MCMC script to create it.")

with open("tess_params.json", "r") as f_json:
    params_dict = json.load(f_json)

# Use cached parameter values
t0_rel = params_dict["t0_rel"]
rp = params_dict["rp"]
a = params_dict["a"]
inc = params_dict["inc"]

print("\nLoaded parameters from tess_params.json:")
print(f"  t0_est: {t0_est}")
print(f"  t0_rel: {t0_rel}")
print(f"  rp: {rp}")
print(f"  a: {a}")
print(f"  inc: {inc}")

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
    m = batman.TransitModel(params, t_rel, supersample_factor=5,
                            exp_time=(t_rel[1]-t_rel[0] if len(t_rel)>1 else 0.002))
    return m.light_curve(params)

best = np.array([t0_rel, rp, a, inc])
best_flux = model(best, t - t0_est)
flux_no_transit = f - best_flux + 1.0

print(f"\nStd of flux_no_transit: {np.std(flux_no_transit - 1.0):.6f}")

phase_full = ((t - (t0_est + t0_rel)) / period) % 1.0
phase_full = np.where(phase_full > 0.5, phase_full - 1, phase_full)

in_transit_mask = np.abs(phase_full) < 0.05
out_transit_mask = ~in_transit_mask

t_in = t[in_transit_mask]
f_in = flux_no_transit[in_transit_mask]

t_out = t[out_transit_mask]
f_out = flux_no_transit[out_transit_mask]

print(f"In-transit points: {len(t_in)}")
print(f"Out-of-transit points: {len(t_out)}")

print(f"\nResidual stats:")
print(f"Full data: mean = {np.mean(flux_no_transit - 1.0):.6e}, std = {np.std(flux_no_transit - 1.0):.6e}")
print(f"Out-of-transit: mean = {np.mean(f_out - 1.0):.6e}, std = {np.std(f_out - 1.0):.6e}")
print(f"In-transit: mean = {np.mean(f_in - 1.0):.6e}, std = {np.std(f_in - 1.0):.6e}")

fig_debug, (ax_out, ax_in) = plt.subplots(2, 1, figsize=(12, 6))
ax_out.plot(t_out, f_out, 'g.', ms=1, alpha=0.3)
ax_out.axhline(1.0, color='r', linestyle='--')
ax_out.set_ylabel('Flux (out-of-transit)')
ax_out.set_title('Out-of-transit residuals')

ax_in.plot(t_in, f_in, 'b.', ms=1, alpha=0.3)
ax_in.axhline(1.0, color='r', linestyle='--')
ax_in.set_xlabel('Time')
ax_in.set_ylabel('Flux (in-transit)')
ax_in.set_title('In-transit residuals')
plt.tight_layout()
plt.show()

orbital_freq = 1.0 / period
print("\nComputing Lomb-Scargle periodograms...")

# Full residuals
ls_full = LombScargle(t, flux_no_transit)
freq_full, power_full = ls_full.autopower(minimum_frequency=0.01, maximum_frequency=20.0, samples_per_peak=10)
probabilities = [0.0455, 0.0027, 6.334e-5, 5.733e-7]
alarm_full = ls_full.false_alarm_level(probabilities)

# Out-of-transit
ls_out = LombScargle(t_out, f_out)
freq_out, power_out = ls_out.autopower(minimum_frequency=0.01, maximum_frequency=20.0, samples_per_peak=10)
alarm_out = ls_out.false_alarm_level(probabilities)

# In-transit
ls_in = LombScargle(t_in, f_in)
freq_in, power_in = ls_in.autopower(minimum_frequency=0.01, maximum_frequency=20.0, samples_per_peak=10)
alarm_in = ls_in.false_alarm_level(probabilities)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11))

def plot_fap_lines(ax, alarm_array):
    labels = ['~10% FAP', '~1% FAP', '~0.01% FAP', '~0.0001% FAP']
    colors = ['red', 'orange', 'purple', 'darkred']
    for amp, label, color in zip(alarm_array, labels, colors):
        ax.axhline(amp, color=color, linestyle='--', alpha=0.5, label=label)

ax1.plot(freq_full, power_full, 'purple', lw=1)
plot_fap_lines(ax1, alarm_full)
ax1.axvline(orbital_freq, color='r', linestyle='--', lw=2, label='Orbital freq')
for n in range(2, 12):
    ax1.axvline(n * orbital_freq, color='orange', linestyle=':', lw=0.8, alpha=0.5)
ax1.set_ylabel('Power')
ax1.set_title('All residuals')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(freq_out, power_out, 'g', lw=1)
plot_fap_lines(ax2, alarm_out)
ax2.axvline(orbital_freq, color='r', linestyle='--', lw=2, label='Orbital freq')
for n in range(2, 12):
    ax2.axvline(n * orbital_freq, color='orange', linestyle=':', lw=0.8, alpha=0.5)
ax2.set_ylabel('Power')
ax2.set_title('Out-of-transit residuals')
ax2.legend()
ax2.grid(alpha=0.3)

ax3.plot(freq_in, power_in, 'blue', lw=1)
plot_fap_lines(ax3, alarm_in)
ax3.axvline(orbital_freq, color='r', linestyle='--', lw=2, label='Orbital freq')
for n in range(2, 12):
    ax3.axvline(n * orbital_freq, color='orange', linestyle=':', lw=0.8, alpha=0.5)
ax3.set_xlabel('Frequency (day⁻¹)')
ax3.set_ylabel('Power')
ax3.set_title('In-transit residuals')
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.show()