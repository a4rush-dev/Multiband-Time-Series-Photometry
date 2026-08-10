import numpy as np
import matplotlib.pyplot as plt
import batman
import emcee
from astropy.timeseries import LombScargle

np.random.seed(42)

# -------------------------
# Load HST data
# -------------------------
INPUT_TXT = "full_LC_stel_puls_orb_params.txt"

data = np.loadtxt(INPUT_TXT)
time = data[0,:]
flux = data[1,:]
flux_err = data[2,:]

t_rel = time - time[0]
flux_norm = flux + 1.0

print(f"HST data: {len(time)} points over {t_rel[-1]:.2f} days")

# -------------------------
# Inflate errors if needed
# -------------------------
out_of_transit_std = np.std(flux_norm[t_rel > 0.3])
if out_of_transit_std > 3 * np.median(flux_err):
    print(f"Inflating errors to match scatter")
    flux_err = np.sqrt(flux_err**2 + (out_of_transit_std * 0.5)**2)

# -------------------------
# Planet flux model
# -------------------------
def planet_flux_model(t, Fp_Fsmin, c1, c2, c3, c4, c5, c6):
    u = np.where(t < c2, (t - c2) / c3, (t - c2) / c4)
    eclipse = (t > (c5 - 0.5*c6)) & (t < (c5 + 0.5*c6))
    model = np.where(eclipse, 0.0, Fp_Fsmin + c1/(u**2 + 1.0))
    return model

# -------------------------
# Combined batman + planet flux model
# -------------------------
period = 5.6334729
orbital_freq = 1.0 / period

base_params = batman.TransitParams()
base_params.per = period
base_params.ecc = 0.516
base_params.w = 188.0
base_params.limb_dark = "quadratic"

def model(theta, t_arr):
    t0, rp, a, inc, u1, u2, Fp_Fsmin, c1, c2, c3, c4, c5, c6 = theta
    
    base_params.t0 = t0
    base_params.rp = rp
    base_params.a = a
    base_params.inc = inc
    base_params.u = [u1, u2]
    
    m = batman.TransitModel(base_params, t_arr, supersample_factor=3, exp_time=0.002)
    transit = m.light_curve(base_params)
    
    planet = planet_flux_model(t_arr, Fp_Fsmin, c1, c2, c3, c4, c5, c6)
    
    return transit + planet - Fp_Fsmin

# -------------------------
# MCMC functions
# -------------------------
def log_likelihood(theta, t_arr, f_arr, ferr_arr):
    model_flux = model(theta, t_arr)
    residuals = (f_arr - model_flux) / ferr_arr
    return -0.5 * np.sum(residuals**2 + np.log(2*np.pi*ferr_arr**2))

def log_prior(theta):
    t0, rp, a, inc, u1, u2, Fp_Fsmin, c1, c2, c3, c4, c5, c6 = theta
    if (0.0 < t0 < 0.3 and 
        0.05 < rp < 0.09 and 
        7.0 < a < 11.0 and 
        84.0 < inc < 89.0 and
        0.0 < u1 < 1.0 and
        0.0 < u2 < 1.0 and
        -0.0001 < Fp_Fsmin < 0.0005 and
        0.0001 < c1 < 0.0015 and
        0.8 < c2 < 1.2 and
        0.5 < c3 < 4.0 and
        0.1 < c4 < 0.5 and
        1.15 < c5 < 1.35 and
        0.05 < c6 < 0.2):
        return 0.0
    return -np.inf

def log_prob(theta, t_arr, f_arr, ferr_arr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t_arr, f_arr, ferr_arr)

# -------------------------
# Run MCMC
# -------------------------
init = np.array([0.14, 0.070, 8.9, 86.3, 0.2, 0.3, 
                 0.00005, 0.0005, 1.05, 1.5, 0.2, 1.23, 0.11])

ndim = 13
nwalkers = 32
nsteps = 3000

pos = init + 1e-5 * np.random.randn(nwalkers, ndim)

print("\nRunning MCMC...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(t_rel, flux_norm, flux_err))
sampler.run_mcmc(pos, nsteps, progress=True)

# -------------------------
# Get best fit
# -------------------------
flat = sampler.get_chain(discard=800, thin=15, flat=True)
best = np.median(flat, axis=0)

best_model = model(best, t_rel)

print(f"\nReduced chi-squared: {np.sum(((flux_norm - best_model) / flux_err)**2) / (len(flux_norm) - ndim):.4f}")

# ===========================
# SIGNAL INJECTION
# ===========================
f_79 = 79 * orbital_freq
f_91 = 91 * orbital_freq

A_79 = 0.000023 # 23 ppm
A_91 = 0.000029 # 29 ppm

n_trials = 5

print(f"\n{'='*60}")
print(f"SIGNAL INJECTION TEST (HST)")
print(f"{'='*60}")
print(f"Injecting signals at:")
print(f"  79th harmonic: f = {f_79:.4f} day^-1 (period = {1/f_79*24:.2f} hours)")
print(f"  91st harmonic: f = {f_91:.4f} day^-1 (period = {1/f_91*24:.2f} hours)")
print(f"  Running {n_trials} trials with random phases")
print(f"{'='*60}\n")

results = []

for trial in range(n_trials):
    print(f"Trial {trial+1}/{n_trials}:")
    
    phi_79 = np.random.uniform(0, 2*np.pi)
    phi_91 = np.random.uniform(0, 2*np.pi)
    
    print(f"  phi_79 = {phi_79:.3f}, phi_91 = {phi_91:.3f}")
    
    signal_79 = A_79 * np.sin(2*np.pi * f_79 * t_rel + phi_79)
    signal_91 = A_91 * np.sin(2*np.pi * f_91 * t_rel + phi_91)
    
    F_inj = flux_norm + signal_79 + signal_91
    
    F_pulsation_inj = F_inj - best_model + 1.0
    
    ls_inj = LombScargle(t_rel, F_pulsation_inj)
    freq_inj, power_inj = ls_inj.autopower(minimum_frequency=5, maximum_frequency=17, samples_per_peak=20)

    probabilities = [0.0455, 0.0027, 6.334e-5, 5.733e-7]
    alarm_amplitude = ls_inj.false_alarm_level(probabilities)
    
    mask_79 = (freq_inj > f_79 - 0.5) & (freq_inj < f_79 + 0.5)
    mask_91 = (freq_inj > f_91 - 0.5) & (freq_inj < f_91 + 0.5)
    
    if np.any(mask_79):
        peak_79_idx = np.argmax(power_inj[mask_79])
        peak_79_freq = freq_inj[mask_79][peak_79_idx]
        peak_79_power = power_inj[mask_79][peak_79_idx]
    else:
        peak_79_freq, peak_79_power = np.nan, np.nan
    
    if np.any(mask_91):
        peak_91_idx = np.argmax(power_inj[mask_91])
        peak_91_freq = freq_inj[mask_91][peak_91_idx]
        peak_91_power = power_inj[mask_91][peak_91_idx]
    else:
        peak_91_freq, peak_91_power = np.nan, np.nan
    
    print(f"  Recovered 79f: freq = {peak_79_freq:.4f} day^-1, power = {peak_79_power:.4f}")
    print(f"  Recovered 91f: freq = {peak_91_freq:.4f} day^-1, power = {peak_91_power:.4f}")
    
    results.append({
        'trial': trial+1,
        'phi_79': phi_79,
        'phi_91': phi_91,
        'freq_79': peak_79_freq,
        'power_79': peak_79_power,
        'freq_91': peak_91_freq,
        'power_91': peak_91_power,
        'freq_full': freq_inj,
        'power_full': power_inj
    })
    print()

# ===========================
# PLOT LAST TRIAL
# ===========================
last_result = results[-1]

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

ax.plot(last_result['freq_full'], last_result['power_full'], 'k-', lw=1.5, alpha=0.7)
# ADD FAP LINES TO PLOT (NEW CODE)
labels = ['~10% FAP', '~1% FAP', '~0.01% FAP', '~0.0001% FAP']
colors = ['red', 'orange', 'purple', 'darkred']
for amp, label, color in zip(alarm_amplitude, labels, colors):
    ax.axhline(amp, color=color, linestyle='--', alpha=0.5, label=label)

ax.axvline(f_79, color='blue', linestyle='--', lw=2, label=f'Injected 79f ({f_79:.2f} d⁻¹)')
ax.axvline(f_91, color='green', linestyle='--', lw=2, label=f'Injected 91f ({f_91:.2f} d⁻¹)')
if not np.isnan(last_result['freq_79']):
    ax.plot(last_result['freq_79'], last_result['power_79'], 'bo', ms=12, label=f'Recovered 79f')
if not np.isnan(last_result['freq_91']):
    ax.plot(last_result['freq_91'], last_result['power_91'], 'go', ms=12, label=f'Recovered 91f')
ax.set_xlabel('Frequency (day⁻¹)', fontsize=12)
ax.set_ylabel('Lomb-Scargle Power', fontsize=12)
ax.set_title(f'HST Signal Injection Test (Trial {n_trials})')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(5, 17)

plt.tight_layout()
plt.show()

# ===========================
# SUMMARY
# ===========================
print(f"\n{'='*60}")
print("SUMMARY:")
print(f"{'='*60}")
print(f"{'Trial':<8} {'phi_79':<10} {'phi_91':<10} {'Recovered 79f':<18} {'Recovered 91f':<18}")
print(f"{'     ':<8} {'      ':<10} {'      ':<10} {'freq (power)':<18} {'freq (power)':<18}")
print("-" * 60)
for r in results:
    print(f"{r['trial']:<8} {r['phi_79']:<10.3f} {r['phi_91']:<10.3f} {r['freq_79']:.3f} ({r['power_79']:.4f})    {r['freq_91']:.3f} ({r['power_91']:.4f})")

print(f"{'='*60}")
print(f"\nInjected frequencies:")
print(f"  79f = {f_79:.4f} day^-1")
print(f"  91f = {f_91:.4f} day^-1")
