# THIS PROVES 2F FREQUENCY CONTRIBUTION. IT'S INTERESTING CUZ IT IMPLIES ITS TIDALLY LOCKED AT THIS FREQ.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batman
from astropy.timeseries import LombScargle
import json

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

# ========== LOAD TRANSIT PARAMETERS ==========
with open("tess_params.json", "r") as f_json:
    params_dict = json.load(f_json)

t0_est = params_dict["t0_est"]
t0_rel = params_dict["t0_rel"]
rp = params_dict["rp"]
a = params_dict["a"]
inc = params_dict["inc"]

# ========== BUILD TRANSIT MODEL ==========
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
transit_model = model(best, t - t0_est)
residuals = f - transit_model + 1.0

# ========== LOMB-SCARGLE ==========
ls = LombScargle(t, residuals, ferr, center_data=True)
frequency, power = ls.autopower(minimum_frequency=0.01, maximum_frequency=20, samples_per_peak=10)

best_freq = frequency[np.argmax(power)]
print(f"Best frequency: {best_freq:.6f} d⁻¹")

# ========== PHASE DATA & GET MODEL ==========
phase = (best_freq * t) % 1.0
sort_idx = np.argsort(phase)

phase_sorted = phase[sort_idx]
residuals_sorted = residuals[sort_idx]

# Get best-fit model at best frequency
t_phase = np.linspace(0, 1/best_freq, 1000)
y_fit = ls.model(t_phase, best_freq)

# ========== PLOT: JUST PHASED DATA + MODEL ==========
fig, ax = plt.subplots(figsize=(8, 6))

# Blue dots: phased data
ax.plot(phase_sorted, residuals_sorted, '.', color='C0', markersize=3, alpha=0.6)

# Black curve: best-fit model
ax.plot(t_phase * best_freq, y_fit, 'k-', lw=2)

ax.set_xlabel('Phase', fontsize=12)
ax.set_ylabel('Normalized Flux', fontsize=12)
ax.set_title(f'Phased Data + Best-Fit Model (f = {best_freq:.6f} d⁻¹)', fontsize=13)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('tess_phased_model.png', dpi=300)
plt.show()