import numpy as np
import matplotlib.pyplot as plt
import batman
import emcee
from scipy.stats import skew, kurtosis

# -------------------------
# Load data
# -------------------------
INPUT_TXT = "full_LC_stel_puls_orb_params.txt"

data = np.loadtxt(INPUT_TXT)
time = data[0,:]
flux = data[1,:]
flux_err = data[2,:]

t_rel = time - time[0]
flux_norm = flux + 1.0

# -------------------------
# CHECK THE ERRORS
# -------------------------
print(f"median flux_err: {np.median(flux_err):.2e}")
print(f"std of flux outside transit (t>0.3): {np.std(flux_norm[t_rel > 0.3]):.2e}")

# if the scatter is way bigger than the errors we need to inflate them
out_of_transit_std = np.std(flux_norm[t_rel > 0.3])
if out_of_transit_std > 3 * np.median(flux_err):
    print(f"WARNING: flux scatter ({out_of_transit_std:.2e}) >> flux_err ({np.median(flux_err):.2e})")
    print("inflating errors to match actual scatter")
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
# Batman setup
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
# MCMC setup
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
        -0.0001 < Fp_Fsmin < 0.0005 and  # allow negative
        0.0001 < c1 < 0.0015 and  # force c1 to be bigger
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
# Initial guess with BIGGER c1
# -------------------------
init = np.array([0.14, 0.070, 8.9, 86.3, 0.2, 0.3, 
                 0.00005, 0.0005, 1.05, 1.5, 0.2, 1.23, 0.11])

ndim = 13
nwalkers = 32
nsteps = 3000

pos = init + 1e-5 * np.random.randn(nwalkers, ndim)

print("\nrunning mcmc with adjusted errors...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(t_rel, flux_norm, flux_err))
sampler.run_mcmc(pos, nsteps, progress=True)

# -------------------------
# Results
# -------------------------
flat = sampler.get_chain(discard=800, thin=15, flat=True)
best = np.median(flat, axis=0)

best_model = model(best, t_rel)
residuals = flux_norm - best_model
chi2 = np.sum((residuals / flux_err)**2)
reduced_chi2 = chi2 / (len(flux_norm) - ndim)

res_skew = skew(residuals / flux_err)
res_kurt = kurtosis(residuals / flux_err)

# -------------------------
# Plot
# -------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})

ax1.errorbar(t_rel, flux_norm, yerr=flux_err, fmt='.k', ms=3, alpha=0.4, label='HST data')
ax1.plot(t_rel, best_model, 'r-', lw=2, label='best fit')
ax1.set_ylabel('Normalized flux')
ax1.legend()
ax1.set_title(f'reduced χ² = {reduced_chi2:.3f} | skew = {res_skew:.3f} | kurtosis = {res_kurt:.3f}')

ax2.errorbar(t_rel, residuals, yerr=flux_err, fmt='.k', ms=3, alpha=0.5)
ax2.axhline(0, color='r', linestyle='--', lw=1)
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Residuals')

plt.tight_layout()
plt.show()

print(f"\nFit statistics:")
print(f"Reduced chi-squared: {reduced_chi2:.4f}")
print(f"Residual skewness: {res_skew:.4f}")
print(f"Residual kurtosis: {res_kurt:.4f}")
print(f"\nBest fit parameters:")
labels = ['t0', 'rp', 'a', 'inc', 'u1', 'u2', 'Fp_Fsmin', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6']
for label, val in zip(labels, best):
    print(f"{label} = {val:.6f}")
