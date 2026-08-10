# import numpy as np
# import matplotlib.pyplot as plt
# import batman
# import emcee

# # ===========================
# # LOAD HST DATA
# # ===========================
# INPUT_TXT = "full_LC_stel_puls_orb_params.txt"

# data = np.loadtxt(INPUT_TXT)
# time = data[0,:]
# flux = data[1,:]
# flux_err = data[2,:]

# t_rel = time - time[0]  # relative time
# flux_norm = flux + 1.0  # normalize around 1.0

# print(f"HST data: {len(time)} points over {t_rel[-1]:.2f} days")

# # ===========================
# # PARAMETERS
# # ===========================
# period = 5.6334729
# orbital_freq = 1.0 / period

# # Injection parameters (same as TESS)
# f_79 = 79 * orbital_freq
# f_91 = 91 * orbital_freq
# A_79 = 0.000005  # smaller amplitude for HST (fewer points)
# A_91 = 0.000005
# phi_79 = np.random.uniform(0, 2*np.pi)
# phi_91 = np.random.uniform(0, 2*np.pi)

# print(f"\nInjecting signals:")
# print(f"  79f_orb = {f_79:.3f} day^-1 (P = {24/f_79:.2f} hr)")
# print(f"  91f_orb = {f_91:.3f} day^-1 (P = {24/f_91:.2f} hr)")

# # ===========================
# # INJECT SIGNALS
# # ===========================
# signal_79 = A_79 * np.sin(2*np.pi * f_79 * t_rel + phi_79)
# signal_91 = A_91 * np.sin(2*np.pi * f_91 * t_rel + phi_91)

# flux_injected = flux_norm + signal_79 + signal_91

# # ===========================
# # FIND T0 FOR FOLDING (HST)
# # ===========================
# # HST only covers 1.4 days, so t0 should be close to the transit we see
# ntrial = 500
# trial_t0 = np.linspace(0, 0.3, ntrial)  # search in first 0.3 days
# depths = []

# for t0_test in trial_t0:
#     phase = ((t_rel - t0_test) / period) % 1.0
#     phase = np.where(phase > 0.5, phase - 1, phase)
#     in_transit = np.abs(phase) < 0.02
#     if np.sum(in_transit) > 5:
#         depths.append(np.median(flux_norm[in_transit]))
#     else:
#         depths.append(1.0)

# t0_best = trial_t0[np.argmin(depths)]
# print(f"\nBest t0 for folding: {t0_best:.6f}")

# # ===========================
# # PHASE FOLD
# # ===========================
# phase = ((t_rel - t0_best) / period) % 1.0
# phase = np.where(phase > 0.5, phase - 1, phase)

# sort_idx = np.argsort(phase)

# # ===========================
# # CREATE BATMAN TRANSIT MODEL (for task 4)
# # ===========================
# print("\nRunning MCMC to get best-fit transit model...")

# # Setup (same as before but simpler - just transit, no planet flux)
# base_params = batman.TransitParams()
# base_params.t0 = t0_best
# base_params.per = period
# base_params.ecc = 0.516
# base_params.w = 188.0
# base_params.limb_dark = "quadratic"

# def transit_model(theta, t_arr):
#     rp, a, inc, u1, u2 = theta
#     base_params.rp = rp
#     base_params.a = a
#     base_params.inc = inc
#     base_params.u = [u1, u2]
    
#     m = batman.TransitModel(base_params, t_arr, supersample_factor=3, exp_time=0.002)
#     return m.light_curve(base_params)

# def log_likelihood(theta, t_arr, f_arr, ferr_arr):
#     model_flux = transit_model(theta, t_arr)
#     residuals = (f_arr - model_flux) / ferr_arr
#     return -0.5 * np.sum(residuals**2)

# def log_prior(theta):
#     rp, a, inc, u1, u2 = theta
#     if (0.05 < rp < 0.09 and 7.0 < a < 11.0 and 84.0 < inc < 89.0 and 
#         0.0 < u1 < 1.0 and 0.0 < u2 < 1.0):
#         return 0.0
#     return -np.inf

# def log_prob(theta, t_arr, f_arr, ferr_arr):
#     lp = log_prior(theta)
#     if not np.isfinite(lp):
#         return -np.inf
#     return lp + log_likelihood(theta, t_arr, f_arr, ferr_arr)

# # Run MCMC
# init = np.array([0.070, 8.9, 86.5, 0.2, 0.3])
# ndim = 5
# nwalkers = 20
# nsteps = 1000

# pos = init + 1e-4 * np.random.randn(nwalkers, ndim)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(t_rel, flux_norm, flux_err))
# sampler.run_mcmc(pos, nsteps, progress=True)

# flat = sampler.get_chain(discard=300, thin=5, flat=True)
# best = np.median(flat, axis=0)

# # Get best-fit model
# best_transit = transit_model(best, t_rel)

# # Pulsations = data - model
# flux_pulsations = flux_norm - best_transit + 1.0

# print(f"\nBest fit transit params: rp={best[0]:.4f}, a={best[1]:.2f}, inc={best[2]:.2f}")

# # ===========================
# # PLOT 1: RAW + INJECTED (UNFOLDED)
# # ===========================
# fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# # Raw
# ax1.plot(t_rel * 24, flux_norm, 'k.', ms=2, alpha=0.5)
# ax1.set_ylabel('Normalized Flux')
# ax1.set_title('Raw HST Data')
# ax1.grid(alpha=0.3)

# # Injected
# ax2.plot(t_rel * 24, flux_injected, 'b.', ms=2, alpha=0.5)
# ax2.set_xlabel('Time (hours)')
# ax2.set_ylabel('Normalized Flux')
# ax2.set_title(f'Injected HST Data (79f + 91f)\n79f period={24/f_79:.2f}hr, 91f period={24/f_91:.2f}hr')
# ax2.grid(alpha=0.3)

# plt.tight_layout()
# # plt.savefig('hst_injected_unfolded.png', dpi=300)
# plt.show()

# # ===========================
# # PLOT 2: PHASE-FOLDED
# # ===========================
# fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# # Raw folded
# ax3.plot(phase[sort_idx], flux_norm[sort_idx], 'k.', ms=2, alpha=0.5)
# ax3.set_ylabel('Normalized Flux')
# ax3.set_title('Phase-Folded Raw HST Data')
# ax3.grid(alpha=0.3)
# ax3.set_xlim(-0.5, 0.5)
# ax3.invert_yaxis()

# # Injected folded
# ax4.plot(phase[sort_idx], flux_injected[sort_idx], 'b.', ms=2, alpha=0.5)
# ax4.set_xlabel('Phase')
# ax4.set_ylabel('Normalized Flux')
# ax4.set_title('Phase-Folded Injected HST Data (79f + 91f)')
# ax4.grid(alpha=0.3)
# ax4.set_xlim(-0.5, 0.5)
# ax4.invert_yaxis()

# plt.tight_layout()
# # plt.savefig('hst_injected_folded.png', dpi=300)
# plt.show()

# # ===========================
# # PLOT 3: HST PULSATIONS (NO INJECTIONS)
# # ===========================
# fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 8))

# # Time domain
# ax5.plot(t_rel * 24, (flux_pulsations - 1.0) * 1e6, 'g.', ms=2, alpha=0.5)
# ax5.axhline(0, color='r', linestyle='--')
# ax5.set_xlabel('Time (hours)')
# ax5.set_ylabel('Pulsations (ppm)')
# ax5.set_title('HST Pulsations (Data - Transit Model)')
# ax5.grid(alpha=0.3)

# # Phase-folded
# ax6.plot(phase[sort_idx], (flux_pulsations[sort_idx] - 1.0) * 1e6, 'g.', ms=2, alpha=0.5)
# ax6.axhline(0, color='r', linestyle='--')
# ax6.set_xlabel('Phase')
# ax6.set_ylabel('Pulsations (ppm)')
# ax6.set_title('Phase-Folded HST Pulsations')
# ax6.grid(alpha=0.3)
# ax6.set_xlim(-0.5, 0.5)

# plt.tight_layout()
# # plt.savefig('hst_pulsations.png', dpi=300)
# plt.show()

# print("\nDone! Created 3 plots:")
# print("  1. hst_injected_unfolded.png")
# print("  2. hst_injected_folded.png")
# print("  3. hst_pulsations.png")
import numpy as np
import matplotlib.pyplot as plt
import batman
import emcee

# ===========================
# LOAD HST DATA
# ===========================
INPUT_TXT = "full_LC_stel_puls_orb_params.txt"

data = np.loadtxt(INPUT_TXT)
time = data[0,:]
flux = data[1,:]
flux_err = data[2,:]

t_rel = time - time[0]  # relative time
flux_norm = flux + 1.0  # normalize around 1.0

print(f"HST data: {len(time)} points over {t_rel[-1]:.2f} days")

# ===========================
# PARAMETERS
# ===========================
period = 5.6334729
orbital_freq = 1.0 / period

# Injection parameters
f_79 = 79 * orbital_freq  # ~14.02 day^-1
f_91 = 91 * orbital_freq  # ~16.15 day^-1
A_79 = 0.001
A_91 = 0.001
phi_79 = np.random.uniform(0, 2*np.pi)
phi_91 = np.random.uniform(0, 2*np.pi)

# BEAT FREQUENCY PERIOD
f_beat = np.abs(f_79 - f_91)  # difference frequency
period_beat = 1.0 / f_beat # F_raw % period_beat

print(f"\nOrbital period: {period:.3f} days")
print(f"Beat frequency: {f_beat:.3f} day^-1")
print(f"Beat period: {period_beat:.3f} days ({period_beat*24:.2f} hours)")
print(f"\nInjecting signals:")
print(f"  79f_orb = {f_79:.3f} day^-1 (P = {24/f_79:.2f} hr)")
print(f"  91f_orb = {f_91:.3f} day^-1 (P = {24/f_91:.2f} hr)")

# ===========================
# INJECT SIGNALS
# ===========================
signal_79 = A_79 * np.sin(2*np.pi * f_79 * t_rel + phi_79)
signal_91 = A_91 * np.sin(2*np.pi * f_91 * t_rel + phi_91)

flux_injected = flux_norm + signal_79 + signal_91

# ===========================
# FIND T0 FOR ORBITAL FOLDING (for transit model)
# ===========================
ntrial = 500
trial_t0 = np.linspace(0, 0.3, ntrial)
depths = []

for t0_test in trial_t0:
    phase = ((t_rel - t0_test) / period) % 1.0
    phase = np.where(phase > 0.5, phase - 1, phase)
    in_transit = np.abs(phase) < 0.02
    if np.sum(in_transit) > 5:
        depths.append(np.median(flux_norm[in_transit]))
    else:
        depths.append(1.0)

t0_best = trial_t0[np.argmin(depths)]
print(f"\nBest t0 for transit: {t0_best:.6f}")

# ===========================
# PLANET FLUX MODEL
# ===========================
def planet_flux_model(t, Fp_Fsmin, c1, c2, c3, c4, c5, c6):
    u = np.where(t < c2, (t - c2) / c3, (t - c2) / c4)
    eclipse = (t > (c5 - 0.5*c6)) & (t < (c5 + 0.5*c6))
    model = np.where(eclipse, 0.0, Fp_Fsmin + c1/(u**2 + 1.0))
    return model

# ===========================
# COMBINED TRANSIT + PLANET FLUX MODEL
# ===========================
base_params = batman.TransitParams()
base_params.per = period
base_params.ecc = 0.516
base_params.w = 188.0
base_params.limb_dark = "quadratic"

def combined_model(theta, t_arr, t0):
    rp, a, inc, u1, u2, Fp_Fsmin, c1, c2, c3, c4, c5, c6 = theta
    
    # Transit
    base_params.t0 = t0
    base_params.rp = rp
    base_params.a = a
    base_params.inc = inc
    base_params.u = [u1, u2]
    
    m = batman.TransitModel(base_params, t_arr, supersample_factor=3, exp_time=0.002)
    transit = m.light_curve(base_params)
    
    # Planet flux
    planet = planet_flux_model(t_arr, Fp_Fsmin, c1, c2, c3, c4, c5, c6)
    
    return transit + planet - Fp_Fsmin

def log_likelihood(theta, t_arr, t0, f_arr, ferr_arr):
    model_flux = combined_model(theta, t_arr, t0)
    residuals = (f_arr - model_flux) / ferr_arr
    return -0.5 * np.sum(residuals**2)

def log_prior(theta):
    rp, a, inc, u1, u2, Fp_Fsmin, c1, c2, c3, c4, c5, c6 = theta
    if (0.05 < rp < 0.09 and 
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

def log_prob(theta, t_arr, t0, f_arr, ferr_arr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t_arr, t0, f_arr, ferr_arr)

# ===========================
# RUN MCMC
# ===========================
print("\nRunning MCMC to fit transit + planet flux model...")

init = np.array([0.070, 8.9, 86.5, 0.2, 0.3, 
                 0.00005, 0.0005, 1.05, 1.5, 0.2, 1.23, 0.11])
ndim = 12
nwalkers = 24
nsteps = 2000

pos = init + 1e-5 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(t_rel, t0_best, flux_norm, flux_err))
sampler.run_mcmc(pos, nsteps, progress=True)

flat = sampler.get_chain(discard=600, thin=10, flat=True)
best = np.median(flat, axis=0)

# Get best-fit combined model
best_model = combined_model(best, t_rel, t0_best)

# Pulsations = data - model
flux_pulsations = flux_norm - best_model + 1.0

print(f"\nBest fit: rp={best[0]:.4f}, a={best[1]:.2f}, inc={best[2]:.2f}")
print(f"          c1={best[6]:.6f}, c2={best[7]:.3f}, c5={best[10]:.3f}")

# ===========================
# PHASE CALCULATIONS
# ===========================
# Orbital phase (for reference)
phase_orbital = ((t_rel - t0_best) / period) % 1.0
phase_orbital = np.where(phase_orbital > 0.5, phase_orbital - 1, phase_orbital)

# BEAT PHASE (for folding plots)
phase_beat = (t_rel % period_beat) / period_beat  # 0 to 1
phase_beat = np.where(phase_beat > 0.5, phase_beat - 1, phase_beat)  # -0.5 to 0.5

# Sort by beat phase for plotting
sort_idx_beat = np.argsort(phase_beat)

# ===========================
# PLOT 1: RAW + INJECTED (UNFOLDED)
# ===========================
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Raw
ax1.plot(t_rel * 24, flux_norm, 'k.', ms=2, alpha=0.5)
ax1.set_ylabel('Normalized Flux')
ax1.set_title('Raw HST Data')
ax1.grid(alpha=0.3)

# Injected
ax2.plot(t_rel * 24, flux_injected, 'b.', ms=2, alpha=0.5)
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Normalized Flux')
ax2.set_title(f'Injected HST Data (79f + 91f)\n79f period={24/f_79:.2f}hr, 91f period={24/f_91:.2f}hr, Beat period={period_beat*24:.2f}hr')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===========================
# PLOT 2: PHASE-FOLDED TO BEAT PERIOD
# ===========================
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Raw folded
ax3.plot(phase_beat[sort_idx_beat], flux_norm[sort_idx_beat], 'k.', ms=2, alpha=0.5)
ax3.set_ylabel('Normalized Flux')
ax3.set_title(f'Phase-Folded Raw HST Data (Beat Period = {period_beat*24:.2f} hours)')
ax3.grid(alpha=0.3)
ax3.set_xlim(-0.5, 0.5)

# Injected folded
ax4.plot(phase_beat[sort_idx_beat], flux_injected[sort_idx_beat], 'b.', ms=2, alpha=0.5)
ax4.set_xlabel(f'Phase (Beat Period = {period_beat:.3f} days)')
ax4.set_ylabel('Normalized Flux')
ax4.set_title('Phase-Folded Injected HST Data (79f + 91f)')
ax4.grid(alpha=0.3)
ax4.set_xlim(-0.5, 0.5)

plt.tight_layout()
plt.show()

# ===========================
# PLOT 3: HST PULSATIONS (NO INJECTIONS)
# ===========================
fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 8))

# Time domain
ax5.plot(t_rel * 24, (flux_pulsations - 1.0) * 1e6, 'g.', ms=2, alpha=0.5)
ax5.axhline(0, color='r', linestyle='--')
ax5.set_xlabel('Time (hours)')
ax5.set_ylabel('Pulsations (ppm)')
ax5.set_title('HST Pulsations (Data - Transit Model - Planet Flux)')
ax5.grid(alpha=0.3)

# Phase-folded to BEAT PERIOD
ax6.plot(phase_beat[sort_idx_beat], (flux_pulsations[sort_idx_beat] - 1.0) * 1e6, 'g.', ms=2, alpha=0.5)
ax6.axhline(0, color='r', linestyle='--')
ax6.set_xlabel(f'Phase (Beat Period = {period_beat:.3f} days)')
ax6.set_ylabel('Pulsations (ppm)')
ax6.set_title(f'Phase-Folded HST Pulsations (Beat Period = {period_beat*24:.2f} hours)')
ax6.grid(alpha=0.3)
ax6.set_xlim(-0.5, 0.5)

plt.tight_layout()
plt.show()