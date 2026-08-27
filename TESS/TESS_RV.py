# TODO: generate a json for mcmc best-fit parameters to feed to LS/spectral.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import batman
import emcee
from scipy import signal

# RV MODULE IMPORT HERE
# NOTE: hat-p-2c or quadratic terms are intentionally not included

params = batman.TransitParams()
params.t0 = 0.0
params.per = 5.6334729
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

    m = batman.TransitModel(
        params,
        t_rel,
        supersample_factor=5,
        exp_time=(t_rel[1] - t_rel[0] if len(t_rel) > 1 else 0.002),
    )

    return m.light_curve(params)


def log_likelihood(theta, t_rel, f_rel, ferr_rel):
    model_flux = model(theta, t_rel)

    res = (f_rel - model_flux) / ferr_rel

    return -0.5 * np.sum(res**2 + np.log(2 * np.pi * ferr_rel**2))


def log_prior(theta):
    t0_rel, rp, a, inc = theta

    if (
        -0.5 < t0_rel < 0.5
        and 0.04 < rp < 0.11
        and 5.0 < a < 15.0
        and 80.0 < inc < 90.0
    ):
        return 0.0

    return -np.inf


def log_prob(theta, t_rel, f_rel, ferr_rel):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, t_rel, f_rel, ferr_rel)


TESS_NDIM = 4
RV_NDIM = 8
JOINT_NDIM = TESS_NDIM + RV_NDIM

TESS_SLICE = slice(0, 4)
RV_SLICE = slice(4, 12)


def split_joint_theta(theta_joint):
    theta_joint = np.asarray(theta_joint)

    theta_tess = theta_joint[TESS_SLICE]
    theta_rv = theta_joint[RV_SLICE]

    return theta_tess, theta_rv

params_rv_coupled = batman.TransitParams()
params_rv_coupled.t0 = 0.0
params_rv_coupled.per = 5.6334729
params_rv_coupled.rp = 0.0722
params_rv_coupled.a = 8.9
params_rv_coupled.inc = 86.3
params_rv_coupled.ecc = 0.516
params_rv_coupled.w = 188.0
params_rv_coupled.limb_dark = "quadratic"
params_rv_coupled.u = [0.2, 0.3]


def model_joint_tess(theta_joint, t_tess_rel, rv_model):
    theta_tess, theta_rv = split_joint_theta(theta_joint)

    t0_rel, rp, a, inc = theta_tess

    (
        tc1,
        k1,
        secosw1,
        sesinw1,
        gamma_hires,
        gamma_harpsn,
        ln_jit_hires,
        ln_jit_harpsn,
    ) = theta_rv

    # Convert RV's sqrt(e)-basis into BATMAN's e and omega representation.
    e_value, omega_rad = e_omega_from_se(secosw1, sesinw1)
    omega_deg = np.rad2deg(omega_rad)

    # Shared orbital geometry from RV model.
    params_rv_coupled.per = rv_model.per0
    params_rv_coupled.ecc = e_value
    params_rv_coupled.w = omega_deg

    # TESS parameters.
    params_rv_coupled.t0 = t0_rel
    params_rv_coupled.rp = rp
    params_rv_coupled.a = a
    params_rv_coupled.inc = inc

    m = batman.TransitModel(
        params_rv_coupled,
        t_tess_rel,
        supersample_factor=5,
        exp_time=(
            t_tess_rel[1] - t_tess_rel[0]
            if len(t_tess_rel) > 1
            else 0.002
        ),
    )

    return m.light_curve(params_rv_coupled)


def log_likelihood_joint_tess(
    theta_joint,
    t_tess_rel,
    f_tess,
    ferr_tess,
    rv_model,
):
    model_flux = model_joint_tess(
        theta_joint,
        t_tess_rel,
        rv_model,
    )

    residuals = (f_tess - model_flux) / ferr_tess

    return -0.5 * np.sum(
        residuals**2
        + np.log(2.0 * np.pi * ferr_tess**2)
    )

def log_prior_joint(theta_joint, rv_model):
    theta_tess, theta_rv = split_joint_theta(theta_joint)

    lp_tess = log_prior(theta_tess)
    if not np.isfinite(lp_tess):
        return -np.inf

    lp_rv = rv_model.log_prior(theta_rv)
    if not np.isfinite(lp_rv):
        return -np.inf

    return lp_tess + lp_rv

def log_likelihood_joint(
    theta_joint,
    t_tess_rel,
    f_tess,
    ferr_tess,
    rv_model,
):
    theta_tess, theta_rv = split_joint_theta(theta_joint)

    lnlike_rv = rv_model.log_likelihood(theta_rv)
    if not np.isfinite(lnlike_rv):
        return -np.inf

    lnlike_tess = log_likelihood_joint_tess(
        theta_joint,
        t_tess_rel,
        f_tess,
        ferr_tess,
        rv_model,
    )
    if not np.isfinite(lnlike_tess):
        return -np.inf

    return lnlike_rv + lnlike_tess


def log_prob_joint(
    theta_joint,
    t_tess_rel,
    f_tess,
    ferr_tess,
    rv_model,
):
    lp = log_prior_joint(theta_joint, rv_model)
    if not np.isfinite(lp):
        return -np.inf

    lnlike = log_likelihood_joint(
        theta_joint,
        t_tess_rel,
        f_tess,
        ferr_tess,
        rv_model,
    )
    if not np.isfinite(lnlike):
        return -np.inf

    return lp + lnlike


def initial_joint_theta(rv_model):
    tess_initial = np.array([
        0.0,      # original TESS t0_rel
        0.0722,   # rp
        8.9,      # a
        86.3,     # inc
    ])

    rv_initial = rv_model.theta_from_params()

    return np.concatenate([tess_initial, rv_initial])

def tess_absolute_time_from_rv(
    t_tess_input,
    t0_rel,
    tc1,
    period,
    tess_time_offset,
):
    t_tess_bjd_tdb = np.asarray(t_tess_input) + tess_time_offset
    return t_tess_bjd_tdb, tc1 + t0_rel


def main() -> None:
    # load
    df = pd.read_csv("hatp2_tess_lightcurve.csv")

    t = np.array(df["time"])
    f = np.array(df["flux"])
    ferr = np.array(df["flux_err"])

    if np.any(~np.isfinite(ferr)) or np.any(ferr <= 0):
        ferr = np.ones_like(f) * np.std(f) * 1.0
        print(
            "warning: flux_err had bad values "
            "-> replaced with global scatter estimate"
        )

    median_flux = np.median(f)
    f = f / median_flux
    ferr = ferr / median_flux

    period = 5.6334729

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

    phase = ((t - t0_est) / period) % 1.0
    phase = phase - np.round(phase)
    window = 0.05
    mask_window = np.abs(phase) < window

    t_win = t[mask_window] - t0_est
    f_win = f[mask_window]
    ferr_win = ferr[mask_window]

    order = np.argsort(t_win)
    t_win = t_win[order]
    f_win = f_win[order]
    ferr_win = ferr_win[order]

    ndim = 4
    nwalkers = 24

    init = np.array([0.0, 0.0722, 8.9, 86.3])
    pos = init + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob,
        args=(t_win, f_win, ferr_win),
    )

    print("running short test (1000 steps)...")
    sampler.run_mcmc(pos, 1000, progress=True)

    flat = sampler.get_chain(discard=200, thin=5, flat=True)

    best = np.median(flat, axis=0)
    t0_rel, rp, a, inc = best

    print("\n" + "=" * 50)
    print("BEST FIT PARAMETERS FROM TESS")
    print("=" * 50)
    print(f"t0_rel (relative) = {t0_rel:.6f}")
    print(f"t0_absolute = {t0_est + t0_rel:.6f}")
    print(f"rp = {rp:.6f}")
    print(f"a = {a:.6f}")
    print(f"inc = {inc:.6f}")
    print(f"period = {period} (fixed)")
    print(f"ecc = {params.ecc} (fixed)")
    print(f"w = {params.w} (fixed)")
    print(f"u = {params.u} (fixed)")
    print("=" * 50)

    best_flux = model(best, t - t0_est)

    plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.errorbar(
        t,
        f,
        yerr=ferr,
        fmt=".k",
        ms=2,
        alpha=0.3,
        label="TESS data",
    )
    ax1.plot(t, best_flux, "r", lw=2, label="best-fit model")
    ax1.set_ylabel("Normalized flux")
    ax1.invert_yaxis()
    ax1.legend()

    rv_model = HatP2RVModel(rv_file="hat_p2_rv.txt")

    print("\n" + "=" * 70)
    print("RV INTEGRATION: INITIALIZATION")
    print("=" * 70)
    print("RV times are treated as BJD_TDB - 2450000 in the input file.")
    print("The modular RV model internally uses absolute BJD_TDB.")
    print(f"RV reference period = {rv_model.per0:.10f} days")
    print(f"RV reference tc1 = {rv_model.tc0:.6f} BJD_TDB")
    print("=" * 70)

    joint_init = initial_joint_theta(rv_model)

    joint_ndim = JOINT_NDIM
    joint_nwalkers = 48

    joint_jump = np.array([
        1.0e-4,  # TESS t0_rel
        1.0e-4,  # TESS rp
        1.0e-2,  # TESS a
        1.0e-2,  # TESS inc
        1.0e-3,  # RV tc1
        5.0,     # RV k1
        1.0e-3,  # RV secosw1
        1.0e-3,  # RV sesinw1
        5.0,     # RV gamma_hires
        5.0,     # RV gamma_harpsn
        1.0e-2,  # RV ln_jit_hires
        1.0e-2,  # RV ln_jit_harpsn
    ])

    pos_joint = (
        joint_init
        + joint_jump * np.random.randn(joint_nwalkers, joint_ndim)
    )

    sampler_joint = emcee.EnsembleSampler(
        joint_nwalkers,
        joint_ndim,
        log_prob_joint,
        args=(t_win, f_win, ferr_win, rv_model),
    )

    print("\nRV INTEGRATION: running short joint test...")
    sampler_joint.run_mcmc(pos_joint, 1000, progress=True)

    flat_joint = sampler_joint.get_chain(
        discard=200,
        thin=5,
        flat=True,
    )

    best_joint = np.median(flat_joint, axis=0)

    best_tess_joint, best_rv_joint = split_joint_theta(best_joint)

    (
        best_t0_rel,
        best_rp,
        best_a,
        best_inc,
    ) = best_tess_joint

    (
        best_tc1,
        best_k1,
        best_secosw1,
        best_sesinw1,
        best_gamma_hires,
        best_gamma_harpsn,
        best_ln_jit_hires,
        best_ln_jit_harpsn,
    ) = best_rv_joint

    best_e, best_omega_rad = e_omega_from_se(
        best_secosw1,
        best_sesinw1,
    )
    best_omega_deg = np.rad2deg(best_omega_rad)

    rv_model.update_params_from_theta(best_rv_joint)
    rv_diag = rv_model.compute_diagnostics()

    print("\nRV DIAGNOSTICS (joint solution)")
    print(f"chi2 = {rv_diag['chi2']:.2f}")
    print(f"chi2_r = {rv_diag['chi2r']:.3f}")
    print(f"RMS_all = {rv_diag['rms']:.2f} m/s")
    print(f"e = {rv_diag['e']:.4f}")
    print(f"omega (deg) = {rv_diag['omega_deg']:.2f}")

    print("\n" + "=" * 70)
    print("BEST FIT PARAMETERS FROM JOINT TESS + RV MODEL")
    print("=" * 70)

    print("\nTESS PARAMETERS")
    print(f"t0_rel = {best_t0_rel:.8f} days")
    print(f"rp = {best_rp:.8f}")
    print(f"a = {best_a:.8f}")
    print(f"inc = {best_inc:.8f} deg")

    print("\nRV PARAMETERS")
    print(f"tc1 = {best_tc1:.8f} BJD_TDB")
    print(f"k1 = {best_k1:.8f} m/s")
    print(f"e = {best_e:.8f}")
    print(f"omega = {best_omega_deg:.8f} deg")
    print(f"gamma_hires = {best_gamma_hires:.8f} m/s")
    print(f"gamma_harpsn = {best_gamma_harpsn:.8f} m/s")
    print(f"jit_hires = {np.exp(best_ln_jit_hires):.8f} m/s")
    print(f"jit_harpsn = {np.exp(best_ln_jit_harpsn):.8f} m/s")

    print("\nSHARED ORBITAL PARAMETERS")
    print(f"period = {rv_model.per0:.10f} days")
    print(f"eccentricity = {best_e:.8f}")
    print(f"omega = {best_omega_deg:.8f} deg")
    print("=" * 70)

    best_flux_joint = model_joint_tess(
        best_joint,
        t_win,
        rv_model,
    )

    fig_joint, ax_joint = plt.subplots(1, 1, figsize=(10, 5))
    ax_joint.errorbar(
        t_win,
        f_win,
        yerr=ferr_win,
        fmt=".k",
        ms=2,
        alpha=0.3,
        label="TESS data",
    )
    ax_joint.plot(
        t_win,
        best_flux_joint,
        "r",
        lw=2,
        label="joint TESS + RV model",
    )
    ax_joint.set_xlabel("Time relative to local TESS transit reference [days]")
    ax_joint.set_ylabel("Normalized flux")
    ax_joint.legend()
    plt.show()

    rv_model.plot_phase_rv(theta=best_rv_joint)
    rv_model.plot_residual_histogram(theta=best_rv_joint)


if __name__ == "__main__":
    main()