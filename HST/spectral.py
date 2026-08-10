import numpy as np
import matplotlib.pyplot as plt
import batman
import emcee
from astropy.timeseries import LombScargle

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
flux_no_transit = flux_norm - best_model + 1.0

chi2 = np.sum(((flux_norm - best_model) / flux_err)**2)
reduced_chi2 = chi2 / (len(flux_norm) - ndim)

print(f"\nReduced chi-squared: {reduced_chi2:.4f}")
print(f"Std of flux_no_transit: {np.std(flux_no_transit - 1.0):.6f}")

# -------------------------
# LOMB-SCARGLE on focused frequency range
# -------------------------
orbital_freq = 1.0 / period

print("\nComputing Lomb-Scargle (5-15 day^-1 for stellar pulsations)...")
ls_focused = LombScargle(t_rel, flux_no_transit)
freq_focused, power_focused = ls_focused.autopower(minimum_frequency=5, maximum_frequency=17, samples_per_peak=20)

probabilities = [0.0455, 0.0027, 6.334e-5, 5.733e-7]
alarm_amplitude = ls_focused.false_alarm_level(probabilities)

# -------------------------
# PLOT
# -------------------------
fig, ax = plt.subplots(1, 1, figsize=(14, 6))

ax.plot(freq_focused, power_focused, 'purple', lw=1.5, label='LS periodogram')

# ADD FAP LINES TO PLOT (NEW CODE)
labels = ['~10% FAP', '~1% FAP', '~0.01% FAP', '~0.0001% FAP']
colors = ['red', 'orange', 'purple', 'darkred']
for amp, label, color in zip(alarm_amplitude, labels, colors):
    ax.axhline(amp, color=color, linestyle='--', alpha=0.5, label=label)

ax.axvline(orbital_freq, color='r', linestyle='--', lw=2, alpha=0.5, label=f'Orbital freq ({orbital_freq:.3f} d⁻¹)')
ax.set_xlabel('Frequency (day⁻¹)', fontsize=12)
ax.set_ylabel('Lomb-Scargle Power', fontsize=12)
ax.set_title('HST Stellar Pulsations (2-5 hour periods)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(5, 17)

plt.tight_layout()
plt.show()