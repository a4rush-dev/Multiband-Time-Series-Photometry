from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import batman
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

LOG = logging.getLogger(__name__)

P_ORB = 5.6334675
T0_TRANSIT = 2455288.84969
ECC = 0.51023
OMEGA_DEG = 188.44
INC_DEG = 86.16

STELLAR_DENSITY_CGS = 0.434
G_CGS = 6.67430e-8


def derive_a_over_rs(stellar_density_cgs: float, period_days: float) -> float:
    period_seconds = period_days * 86400.0
    a_rs_cubed = stellar_density_cgs * G_CGS * period_seconds ** 2 / (3.0 * np.pi)
    return float(a_rs_cubed ** (1.0 / 3.0))


A_RS = derive_a_over_rs(STELLAR_DENSITY_CGS, P_ORB)

TRANSIT_DEPTH_PPM_GUESS = 4941.0
LD_U1_GUESS = 0.06
LD_U2_GUESS = 0.23

EXPOSURE_SECONDS = 0.4
EXPOSURE_DAYS = EXPOSURE_SECONDS / 86400.0
BATMAN_SUPERSAMPLE = 7

DEFAULT_BIN_WIDTH = 0.00025

N_IP_NEIGHBORS = 50
MIN_IP_POINTS = 200
MIN_KERNEL_SIGMA_XY = 0.002
MIN_KERNEL_SIGMA_BETA = 0.02
IP_QUERY_CHUNK_SIZE = 25000
MIN_NEIGHBOR_TIME_SEPARATION_MIN = 2.0

BIN_ERROR_FLOOR_PPM = 25.0

PHASE_WRAP_OFFSET = 0.05

ECLIPSE_EVENT_HALF_WIDTH = 0.010
ECLIPSE_BASELINE_HALF_WIDTH = 0.028
ECLIPSE_DEPTH_PRIOR_SIGMA_PPM = 120.0

DEFAULT_N_WALKERS = 72
DEFAULT_N_STEPS = 9000
DEFAULT_N_BURN = 3500
RANDOM_SEED = 24601

PARAM_NAMES = [
    "depth_ppm", "u1", "u2", "eclipse_depth_ppm",
    "f_min_ppm", "c1_ppm", "t_peak_hr", "tau_rise_hr", "tau_decay_hr",
    "jitter_ppm",
]

PRIOR_BOUNDS = {
    "depth_ppm": (3000.0, 7000.0),
    "u1": (0.0, 1.0),
    "u2": (0.0, 1.0),
    "eclipse_depth_ppm": (0.0, 3000.0),
    "f_min_ppm": (-1000.0, 3000.0),
    "c1_ppm": (0.0, 4000.0),
    "t_peak_hr": (-20.0, 40.0),
    "tau_rise_hr": (0.05, 50.0),
    "tau_decay_hr": (0.05, 50.0),
    "jitter_ppm": (0.0, 4000.0),
}


def mad_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad)


def robust_location(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def as_bool_mask(series):
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "t", "1", "yes", "y"]).to_numpy(dtype=bool)


def phase_fold_01(bjd, period_days=P_ORB, t0_bjd=T0_TRANSIT, wrap_offset=PHASE_WRAP_OFFSET):
    bjd = np.asarray(bjd, dtype=float)
    raw = ((bjd - t0_bjd) / period_days) % 1.0
    return (raw + wrap_offset) % 1.0


def centered_phase(phase_01, center_01):
    phase_01 = np.asarray(phase_01, dtype=float)
    return ((phase_01 - center_01 + 0.5) % 1.0) - 0.5


def circular_phase_distance(phase, center):
    return np.abs(centered_phase(np.asarray(phase, float), center))


def true_to_eccentric_anomaly(true_anomaly, eccentricity):
    tan_half_e = np.sqrt((1.0 - eccentricity) / (1.0 + eccentricity)) * np.tan(0.5 * true_anomaly)
    eccentric_anomaly = 2.0 * np.arctan(tan_half_e)
    return float(eccentric_anomaly % (2.0 * np.pi))


def true_to_mean_anomaly(true_anomaly, eccentricity):
    eccentric_anomaly = true_to_eccentric_anomaly(true_anomaly, eccentricity)
    return float((eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) % (2.0 * np.pi))


def transit_to_periastron_time(t_transit_bjd, period_days, eccentricity, omega_deg):
    omega = np.deg2rad(omega_deg)
    f_transit = 0.5 * np.pi - omega
    m_transit = true_to_mean_anomaly(f_transit, eccentricity)
    return float(t_transit_bjd - period_days * m_transit / (2.0 * np.pi))


def transit_to_occultation_time(t_transit_bjd, period_days, eccentricity, omega_deg):
    omega = np.deg2rad(omega_deg)
    f_transit = 0.5 * np.pi - omega
    f_occultation = 1.5 * np.pi - omega
    m_transit = true_to_mean_anomaly(f_transit, eccentricity)
    m_occultation = true_to_mean_anomaly(f_occultation, eccentricity)
    delta_m = (m_occultation - m_transit) % (2.0 * np.pi)
    return float(t_transit_bjd + period_days * delta_m / (2.0 * np.pi))


T_PERI_REF = transit_to_periastron_time(T0_TRANSIT, P_ORB, ECC, OMEGA_DEG)
T_OCC_REF = transit_to_occultation_time(T0_TRANSIT, P_ORB, ECC, OMEGA_DEG)

PHASE_TRANSIT = phase_fold_01(np.array([T0_TRANSIT]))[0]
PHASE_OCCULTATION = phase_fold_01(np.array([T_OCC_REF]))[0]
PHASE_PERIASTRON = phase_fold_01(np.array([T_PERI_REF]))[0]


def nearest_periodic_epoch(reference_event_bjd, query_times_bjd, period_days=P_ORB):
    query_times_bjd = np.asarray(query_times_bjd, dtype=float)
    epoch_number = np.rint((query_times_bjd - reference_event_bjd) / period_days)
    return reference_event_bjd + epoch_number * period_days


def hours_since_periastron(time_bjd):
    time_bjd = np.asarray(time_bjd, dtype=float)
    t_peri = nearest_periodic_epoch(T_PERI_REF, time_bjd, P_ORB)
    return (time_bjd - t_peri) * 24.0


def make_geometry_params(rp_rs, u1, u2):
    params = batman.TransitParams()
    params.t0 = T0_TRANSIT
    params.per = P_ORB
    params.rp = rp_rs
    params.a = A_RS
    params.inc = INC_DEG
    params.ecc = ECC
    params.w = OMEGA_DEG
    params.t_secondary = T_OCC_REF
    params.limb_dark = "quadratic"
    params.u = [u1, u2]
    params.fp = 0.0
    return params


def compute_transit_shape(time_bjd, rp_rs, u1, u2):
    time_bjd = np.asarray(time_bjd, dtype=float)
    params = make_geometry_params(rp_rs, u1, u2)
    model = batman.TransitModel(
        params, time_bjd, transittype="primary",
        supersample_factor=BATMAN_SUPERSAMPLE, exp_time=EXPOSURE_DAYS,
    )
    return model.light_curve(params)


def compute_eclipse_window(time_bjd, rp_rs):
    time_bjd = np.asarray(time_bjd, dtype=float)
    params = make_geometry_params(rp_rs, 0.0, 0.0)
    params.fp = 1.0
    params.limb_dark = "uniform"
    params.u = []
    model = batman.TransitModel(
        params, time_bjd, transittype="secondary",
        supersample_factor=BATMAN_SUPERSAMPLE, exp_time=EXPOSURE_DAYS,
    )
    unit_secondary_flux = model.light_curve(params)
    visibility = unit_secondary_flux - 1.0
    return 1.0 - visibility


def asymmetric_lorentzian_ppm(hours_since_peri, f_min_ppm, c1_ppm, t_peak_hr, tau_rise_hr, tau_decay_hr):
    t = np.asarray(hours_since_peri, dtype=float)
    tau_rise_hr = max(float(tau_rise_hr), 0.05)
    tau_decay_hr = max(float(tau_decay_hr), 0.05)
    u = np.where(
        t < t_peak_hr,
        (t - t_peak_hr) / tau_rise_hr,
        (t - t_peak_hr) / tau_decay_hr,
    )
    return f_min_ppm + c1_ppm / (u * u + 1.0)


def astrophysical_flux(time_bjd, theta):
    time_bjd = np.asarray(time_bjd, dtype=float)
    (
        depth_ppm, u1, u2, eclipse_depth_ppm,
        f_min_ppm, c1_ppm, t_peak_hr, tau_rise_hr, tau_decay_hr,
        jitter_ppm,
    ) = theta

    rp_rs = np.sqrt(max(float(depth_ppm), 1.0) / 1.0e6)
    transit_shape = compute_transit_shape(time_bjd, rp_rs, u1, u2)
    eclipse_window = compute_eclipse_window(time_bjd, rp_rs)
    hours_since_peri = hours_since_periastron(time_bjd)
    fp_ppm = asymmetric_lorentzian_ppm(hours_since_peri, f_min_ppm, c1_ppm, t_peak_hr, tau_rise_hr, tau_decay_hr)

    return (
        transit_shape
        + fp_ppm * 1.0e-6
        - (eclipse_depth_ppm * 1.0e-6) * eclipse_window
    )


def standardized_coordinates(x_cent, y_cent, beta):
    x_cent = np.asarray(x_cent, dtype=float)
    y_cent = np.asarray(y_cent, dtype=float)
    beta = np.asarray(beta, dtype=float)
    x_scale = max(mad_std(x_cent), MIN_KERNEL_SIGMA_XY)
    y_scale = max(mad_std(y_cent), MIN_KERNEL_SIGMA_XY)
    beta_scale = max(mad_std(beta), MIN_KERNEL_SIGMA_BETA)
    return np.column_stack([
        (x_cent - np.median(x_cent)) / x_scale,
        (y_cent - np.median(y_cent)) / y_scale,
        (beta - np.median(beta)) / beta_scale,
    ])


def leave_one_out_pixel_map(time_bjd, x_cent, y_cent, beta, flux, n_neighbors=N_IP_NEIGHBORS):
    time_bjd = np.asarray(time_bjd, dtype=float)
    x_cent = np.asarray(x_cent, dtype=float)
    y_cent = np.asarray(y_cent, dtype=float)
    beta = np.asarray(beta, dtype=float)
    flux = np.asarray(flux, dtype=float)

    n = flux.size
    if n < MIN_IP_POINTS:
        LOG.warning("Only %d usable points in this AOR; returning unity pixel map.", n)
        return np.ones(n, dtype=float)

    coordinates = standardized_coordinates(x_cent, y_cent, beta)
    tree = cKDTree(coordinates)

    query_k = min(n, max(4 * n_neighbors + 1, n_neighbors + 10))
    _, candidates_all = tree.query(coordinates, k=query_k, workers=-1)
    if candidates_all.ndim == 1:
        candidates_all = candidates_all[:, None]

    minimum_time_days = MIN_NEIGHBOR_TIME_SEPARATION_MIN / (24.0 * 60.0)
    sensitivity = np.ones(n, dtype=float)

    for start in range(0, n, IP_QUERY_CHUNK_SIZE):
        stop = min(start + IP_QUERY_CHUNK_SIZE, n)
        for i in range(start, stop):
            candidate_indices = candidates_all[i]
            candidate_indices = candidate_indices[candidate_indices != i]

            time_separated = np.abs(time_bjd[candidate_indices] - time_bjd[i]) >= minimum_time_days
            candidate_indices = candidate_indices[time_separated]

            candidate_indices = candidate_indices[
                np.isfinite(flux[candidate_indices]) & (flux[candidate_indices] > 0.0)
            ]

            neighbors = candidate_indices[:n_neighbors]
            if neighbors.size < max(10, n_neighbors // 4):
                sensitivity[i] = 1.0
                continue

            delta = coordinates[neighbors] - coordinates[i]
            sigma = np.std(delta, axis=0, ddof=1)
            sigma[0] = max(sigma[0], 0.20)
            sigma[1] = max(sigma[1], 0.20)
            sigma[2] = max(sigma[2], 0.20)

            exponent = -0.5 * np.sum((delta / sigma) ** 2, axis=1)
            weights = np.exp(np.clip(exponent, -700.0, 0.0))

            weight_sum = np.sum(weights)
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                sensitivity[i] = 1.0
                continue

            estimate = np.sum(weights * flux[neighbors]) / weight_sum
            sensitivity[i] = estimate if np.isfinite(estimate) and estimate > 0.0 else 1.0

        LOG.info("Pixel-map progress: %d / %d", stop, n)

    valid = np.isfinite(sensitivity) & (sensitivity > 0.0)
    if not np.any(valid):
        return np.ones(n, dtype=float)

    sensitivity /= np.median(sensitivity[valid])
    return sensitivity


def apply_pixel_map_per_aor(df):
    df = df.copy()
    df["ip_sensitivity"] = 1.0
    df["flux_corr_ipix"] = df["flux_norm_global"].to_numpy(dtype=float)

    if "aor_id" not in df.columns:
        df["aor_id"] = "all"

    for aor_id, group in df.groupby("aor_id", sort=False):
        indices = group.index.to_numpy()
        flux = group["flux_norm_global"].to_numpy(dtype=float)
        time_bjd = group["bjd_utc"].to_numpy(dtype=float)
        x_cent = group["x_cent"].to_numpy(dtype=float)
        y_cent = group["y_cent"].to_numpy(dtype=float)
        beta = group["beta"].to_numpy(dtype=float)

        finite = (
            np.isfinite(time_bjd) & np.isfinite(x_cent) & np.isfinite(y_cent)
            & np.isfinite(beta) & np.isfinite(flux) & (flux > 0.0)
        )

        sensitivity = np.ones(indices.size, dtype=float)

        if finite.sum() >= MIN_IP_POINTS:
            LOG.info("Building pixel map for AOR %s (%d valid frames).", aor_id, int(finite.sum()))
            sensitivity[finite] = leave_one_out_pixel_map(
                time_bjd=time_bjd[finite], x_cent=x_cent[finite],
                y_cent=y_cent[finite], beta=beta[finite], flux=flux[finite],
            )
        else:
            LOG.warning("Skipping pixel map for AOR %s: %d valid frames.", aor_id, int(finite.sum()))

        df.loc[indices, "ip_sensitivity"] = sensitivity
        df.loc[indices, "flux_corr_ipix"] = flux / sensitivity

    return df


def out_of_event_baseline_mask(phase_01):
    phase_01 = np.asarray(phase_01, dtype=float)
    near_transit = circular_phase_distance(phase_01, PHASE_TRANSIT) < 0.035
    near_occultation = circular_phase_distance(phase_01, PHASE_OCCULTATION) < 0.035
    near_periastron = circular_phase_distance(phase_01, PHASE_PERIASTRON) < 0.065
    return ~(near_transit | near_occultation | near_periastron)


def normalize_per_aor_baseline(df):
    df = df.copy()
    df["aor_baseline"] = np.nan
    df["flux_corr_final"] = np.nan

    for aor_id, group in df.groupby("aor_id", sort=False):
        indices = group.index.to_numpy()
        phase = group["phase"].to_numpy(dtype=float)
        flux = group["flux_corr_ipix"].to_numpy(dtype=float)

        finite = np.isfinite(phase) & np.isfinite(flux) & (flux > 0.0)
        baseline_mask = finite & out_of_event_baseline_mask(phase)
        if baseline_mask.sum() < 20:
            baseline_mask = finite

        baseline = robust_location(flux[baseline_mask])
        if not np.isfinite(baseline) or baseline <= 0.0:
            baseline = 1.0

        df.loc[indices, "aor_baseline"] = baseline
        df.loc[indices, "flux_corr_final"] = flux / baseline

    global_scale = robust_location(df["flux_corr_final"].to_numpy(float))
    if np.isfinite(global_scale) and global_scale > 0.0:
        df["flux_corr_final"] /= global_scale

    return df


def bin_phase_curve(phase_01, flux, bin_width=DEFAULT_BIN_WIDTH):
    phase_01 = np.asarray(phase_01, dtype=float)
    flux = np.asarray(flux, dtype=float)

    valid = np.isfinite(phase_01) & np.isfinite(flux) & (phase_01 >= 0.0) & (phase_01 < 1.0)
    phase_01 = phase_01[valid]
    flux = flux[valid]

    n_bins = int(np.round(1.0 / bin_width))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.digitize(phase_01, edges, right=False) - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    rows = []
    for bin_id in range(n_bins):
        use = bin_index == bin_id
        if not np.any(use):
            continue

        phase_here = phase_01[use]
        flux_here = flux[use]
        n_points = flux_here.size

        scatter = mad_std(flux_here)
        if not np.isfinite(scatter) or scatter <= 0.0:
            scatter = np.std(flux_here, ddof=1) if n_points > 1 else np.nan

        error_ppm = scatter * 1.0e6 / np.sqrt(n_points) if np.isfinite(scatter) and n_points > 1 else BIN_ERROR_FLOOR_PPM
        error_ppm = max(error_ppm, BIN_ERROR_FLOOR_PPM)

        rows.append({
            "bin_id": bin_id,
            "phase_left": edges[bin_id],
            "phase_right": edges[bin_id + 1],
            "phase_center": 0.5 * (edges[bin_id] + edges[bin_id + 1]),
            "phase_median": float(np.median(phase_here)),
            "flux_median": float(np.median(flux_here)),
            "flux_mean": float(np.mean(flux_here)),
            "flux_err": error_ppm / 1.0e6,
            "flux_err_ppm": error_ppm,
            "n_points": int(n_points),
        })

    return pd.DataFrame(rows)


def measure_eclipse_depth_ppm(binned):
    phase = binned["phase_center"].to_numpy(dtype=float)
    flux = binned["flux_median"].to_numpy(dtype=float)

    dist = circular_phase_distance(phase, PHASE_OCCULTATION)
    event_mask = dist <= ECLIPSE_EVENT_HALF_WIDTH
    baseline_mask = (dist > ECLIPSE_EVENT_HALF_WIDTH) & (dist <= ECLIPSE_BASELINE_HALF_WIDTH)

    if event_mask.sum() < 5 or baseline_mask.sum() < 10:
        return np.nan

    baseline_level = robust_location(flux[baseline_mask])
    event_level = robust_location(flux[event_mask])

    if not np.isfinite(baseline_level) or not np.isfinite(event_level):
        return np.nan

    return float((baseline_level - event_level) * 1.0e6)


@dataclass(frozen=True)
class FitSummary:
    depth_ppm: float
    u1: float
    u2: float
    eclipse_depth_ppm: float
    f_min_ppm: float
    c1_ppm: float
    t_peak_hr: float
    tau_rise_hr: float
    tau_decay_hr: float
    jitter_ppm: float


def log_prior(theta, eclipse_depth_prior_mean_ppm):
    for value, name in zip(theta, PARAM_NAMES):
        lo, hi = PRIOR_BOUNDS[name]
        if not (lo <= value <= hi):
            return -np.inf

    lp = 0.0
    eclipse_depth_ppm = theta[PARAM_NAMES.index("eclipse_depth_ppm")]
    if np.isfinite(eclipse_depth_prior_mean_ppm):
        lp += -0.5 * ((eclipse_depth_ppm - eclipse_depth_prior_mean_ppm) / ECLIPSE_DEPTH_PRIOR_SIGMA_PPM) ** 2

    return lp


def log_likelihood(theta, time_bjd, flux_binned, err_binned):
    model = astrophysical_flux(time_bjd, theta)
    jitter_ppm = theta[PARAM_NAMES.index("jitter_ppm")]

    sigma2 = err_binned * err_binned + (jitter_ppm * 1.0e-6) ** 2
    resid2 = (flux_binned - model) ** 2

    return float(-0.5 * np.sum(resid2 / sigma2 + np.log(2.0 * np.pi * sigma2)))


def log_probability(theta, time_bjd, flux_binned, err_binned, eclipse_depth_prior_mean_ppm):
    lp = log_prior(theta, eclipse_depth_prior_mean_ppm)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, time_bjd, flux_binned, err_binned)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def preoptimize_start(theta_init, args):
    def neg_log_prob_safe(theta):
        lp = log_probability(theta, *args)
        if not np.isfinite(lp):
            return 1.0e10
        return -lp

    result = minimize(
        neg_log_prob_safe, theta_init, method="Nelder-Mead",
        options={"maxiter": 12000, "xatol": 1.0e-7, "fatol": 1.0e-7},
    )
    theta_start = result.x if result.success else theta_init

    LOG.info("Pre-optimization success=%s", result.success)
    for name, value in zip(PARAM_NAMES, theta_start):
        LOG.info("  start %-18s = %9.3f", name, value)

    return theta_start


def initialize_walkers(n_walkers, theta_start, args, rng):
    spread = np.array([
        200.0,
        0.05,
        0.05,
        60.0,
        25.0,
        30.0,
        0.5,
        0.8,
        3.0,
        25.0,
    ])
    positions = np.empty((n_walkers, theta_start.size), dtype=float)

    for i in range(n_walkers):
        candidate = theta_start + spread * rng.normal(size=theta_start.size)
        attempts = 0
        while not np.isfinite(log_probability(candidate, *args)) and attempts < 150:
            candidate = theta_start + spread * rng.normal(size=theta_start.size)
            attempts += 1
        positions[i] = candidate

    return positions


def run_emcee_fit(binned, n_walkers, n_steps, n_burn, seed):
    if n_burn >= n_steps:
        raise ValueError("--burn must be smaller than --steps.")

    phase = binned["phase_center"].to_numpy(dtype=float)
    time_bjd = T0_TRANSIT + (phase - PHASE_TRANSIT) * P_ORB
    flux = binned["flux_median"].to_numpy(dtype=float)
    flux_err = binned["flux_err"].to_numpy(dtype=float)

    valid = np.isfinite(time_bjd) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
    time_bjd = time_bjd[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]

    if time_bjd.size < 30:
        raise RuntimeError("Too few valid binned points for EMCEE.")

    eclipse_depth_prior_mean_ppm = measure_eclipse_depth_ppm(binned)
    LOG.info(
        "Eclipse-depth prior mean from data: %.1f ppm (prior sigma=%.1f ppm).",
        eclipse_depth_prior_mean_ppm,
        ECLIPSE_DEPTH_PRIOR_SIGMA_PPM,
    )

    args = (time_bjd, flux, flux_err, eclipse_depth_prior_mean_ppm)

    depth_init = TRANSIT_DEPTH_PPM_GUESS
    eclipse_init = (
        eclipse_depth_prior_mean_ppm
        if np.isfinite(eclipse_depth_prior_mean_ppm) and eclipse_depth_prior_mean_ppm > 0.0
        else 900.0
    )

    theta_init = np.array([
        depth_init, LD_U1_GUESS, LD_U2_GUESS, eclipse_init,
        322.0, 856.0, 5.4, 5.5, 10.3, 100.0,
    ])

    LOG.info("Pre-optimizing (BATMAN shapes now recomputed per-evaluation; this is slower)...")
    theta_start = preoptimize_start(theta_init, args)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    p0 = initialize_walkers(n_walkers, theta_start, args, rng)
    sampler = emcee.EnsembleSampler(n_walkers, theta_start.size, log_probability, args=args)

    LOG.info("Running EMCEE: %d walkers, %d steps, %d burn-in.", n_walkers, n_steps, n_burn)
    sampler.run_mcmc(p0, n_steps, progress=True)

    flat_chain = sampler.get_chain(discard=n_burn, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=n_burn, flat=True)

    best_index = int(np.argmax(flat_log_prob))
    theta_map = flat_chain[best_index]

    LOG.info(
        "Best log-probability in chain: %.2f (sample %d of %d).",
        flat_log_prob[best_index],
        best_index,
        flat_chain.shape[0],
    )

    summary = FitSummary(
        depth_ppm=float(theta_map[0]),
        u1=float(theta_map[1]),
        u2=float(theta_map[2]),
        eclipse_depth_ppm=float(theta_map[3]),
        f_min_ppm=float(theta_map[4]),
        c1_ppm=float(theta_map[5]),
        t_peak_hr=float(theta_map[6]),
        tau_rise_hr=float(theta_map[7]),
        tau_decay_hr=float(theta_map[8]),
        jitter_ppm=float(theta_map[9]),
    )

    return summary, flat_chain, flat_log_prob, eclipse_depth_prior_mean_ppm


def fixed_dewit_summary():
    return FitSummary(
        depth_ppm=TRANSIT_DEPTH_PPM_GUESS, u1=LD_U1_GUESS, u2=LD_U2_GUESS,
        eclipse_depth_ppm=900.0, f_min_ppm=322.0, c1_ppm=856.0,
        t_peak_hr=5.40, tau_rise_hr=5.5, tau_decay_hr=10.3, jitter_ppm=100.0,
    )


def fit_summary_to_theta(summary):
    return np.array([
        summary.depth_ppm, summary.u1, summary.u2, summary.eclipse_depth_ppm,
        summary.f_min_ppm, summary.c1_ppm, summary.t_peak_hr,
        summary.tau_rise_hr, summary.tau_decay_hr, summary.jitter_ppm,
    ], dtype=float)


def plot_dewit_fig1_style(binned, fit, output_png):
    theta = fit_summary_to_theta(fit)

    phase_grid = np.linspace(0.0, 1.0, 20001, endpoint=False)
    time_grid = T0_TRANSIT + (phase_grid - PHASE_TRANSIT) * P_ORB
    model_grid = astrophysical_flux(time_grid, theta)

    phase_bin = binned["phase_center"].to_numpy(dtype=float)
    flux_bin = binned["flux_median"].to_numpy(dtype=float)

    baseline_mask = (
        (circular_phase_distance(phase_bin, PHASE_TRANSIT) > 0.04)
        & (circular_phase_distance(phase_bin, PHASE_OCCULTATION) > 0.04)
        & (circular_phase_distance(phase_bin, PHASE_PERIASTRON) > 0.07)
    )

    display_baseline = robust_location(flux_bin[baseline_mask])
    if not np.isfinite(display_baseline) or display_baseline <= 0.0:
        display_baseline = robust_location(flux_bin)

    flux_plot = flux_bin / display_baseline
    model_plot = model_grid / display_baseline

    transit_phase = centered_phase(phase_bin, PHASE_TRANSIT)
    transit_grid = centered_phase(phase_grid, PHASE_TRANSIT)
    occultation_phase = centered_phase(phase_bin, PHASE_OCCULTATION)
    occultation_grid = centered_phase(phase_grid, PHASE_OCCULTATION)

    transit_window = 0.030
    occultation_window = 0.030

    transit_points = np.abs(transit_phase) <= transit_window
    transit_model = np.abs(transit_grid) <= transit_window
    occultation_points = np.abs(occultation_phase) <= occultation_window
    occultation_model = np.abs(occultation_grid) <= occultation_window

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })

    fig = plt.figure(figsize=(9.4, 6.6), dpi=180)
    grid = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.08, 1.0], hspace=0.34, wspace=0.30)

    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    ax_a.plot(
        phase_bin, flux_plot, linestyle="none", marker="o", markersize=2.4,
        color="black", alpha=0.9, rasterized=True, label="Binned photometry",
    )
    ax_a.plot(phase_grid, model_plot, color="#00b300", linewidth=1.6, label="Best-fit model", zorder=4)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_xlabel("Orbital Phase")
    ax_a.set_ylabel(r"$F/F_\star$")
    ax_a.legend(loc="upper right", frameon=False, handlelength=2.4)
    ax_a.text(0.985, 0.94, "A", transform=ax_a.transAxes, ha="right", va="top", fontweight="bold")

    ax_b.plot(
        transit_phase[transit_points], (flux_plot[transit_points] - 1.0) * 1.0e6,
        linestyle="none", marker="o", markersize=2.5, color="black", alpha=0.9, rasterized=True,
    )
    ax_b.plot(
        transit_grid[transit_model], (model_plot[transit_model] - 1.0) * 1.0e6,
        color="#00b300", linewidth=1.6, zorder=4,
    )
    ax_b.set_xlim(-transit_window, transit_window)
    ax_b.set_xlabel("Orbital Phase (centered on transit)")
    ax_b.set_ylabel(r"$(F/F_\star - 1)$ [ppm]")
    ax_b.text(0.95, 0.92, "B", transform=ax_b.transAxes, ha="right", va="top", fontweight="bold")

    ax_c.plot(
        occultation_phase[occultation_points], (flux_plot[occultation_points] - 1.0) * 1.0e6,
        linestyle="none", marker="o", markersize=2.5, color="black", alpha=0.9, rasterized=True,
    )
    ax_c.plot(
        occultation_grid[occultation_model], (model_plot[occultation_model] - 1.0) * 1.0e6,
        color="#00b300", linewidth=1.6, zorder=4,
    )
    ax_c.set_xlim(-occultation_window, occultation_window)
    ax_c.set_xlabel("Orbital Phase (centered on occultation)")
    ax_c.set_ylabel(r"$(F/F_\star - 1)$ [ppm]")
    ax_c.text(0.95, 0.92, "C", transform=ax_c.transAxes, ha="right", va="top", fontweight="bold")

    for axis in (ax_a, ax_b, ax_c):
        axis.grid(alpha=0.12, linewidth=0.5)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    LOG.info("Saved Figure 1-style phase-curve plot: %s", output_png)


def standardize_phase1_columns(df):
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    mapping = {}
    canonical = {
        "globalindex": "global_index", "aorid": "aor_id",
        "visitlabel": "visit_label", "visitindex": "visit_index",
        "segmentid": "segment_id", "bjdutc": "bjd_utc",
        "fluxraw": "flux_raw", "fluxnormvisit": "flux_norm_visit",
        "fluxnormglobal": "flux_norm_global", "xcent": "x_cent",
        "ycent": "y_cent", "frameok": "frame_ok",
    }

    for column in df.columns:
        compact = column.lower().replace("_", "").replace(" ", "")
        if compact in canonical:
            mapping[column] = canonical[compact]

    return df.rename(columns=mapping)


def load_phase1_photometry(input_csv):
    df = pd.read_csv(input_csv)
    df = standardize_phase1_columns(df)

    required = ["bjd_utc", "flux_norm_global", "x_cent", "y_cent", "beta"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Phase 1 CSV is missing required columns: {missing}")

    if "global_index" not in df.columns:
        df["global_index"] = np.arange(df.shape[0], dtype=int)
    if "aor_id" not in df.columns:
        df["aor_id"] = "all"

    if "frame_ok" in df.columns:
        keep = as_bool_mask(df["frame_ok"])
        df = df.loc[keep].copy()

    finite = (
        np.isfinite(df["bjd_utc"].to_numpy(float))
        & np.isfinite(df["flux_norm_global"].to_numpy(float))
        & np.isfinite(df["x_cent"].to_numpy(float))
        & np.isfinite(df["y_cent"].to_numpy(float))
        & np.isfinite(df["beta"].to_numpy(float))
        & (df["flux_norm_global"].to_numpy(float) > 0.0)
    )

    df = df.loc[finite].copy()
    df = df.sort_values(["bjd_utc", "global_index"]).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No usable frames remain after Phase 1 selection.")

    return df


def posterior_summary_dataframe(flat_chain, fit, eclipse_depth_prior_mean_ppm):
    q16, q50, q84 = np.percentile(flat_chain, [16.0, 50.0, 84.0], axis=0)

    data = {}
    for i, name in enumerate(PARAM_NAMES):
        data[f"{name}_p16"] = q16[i]
        data[f"{name}_p50"] = q50[i]
        data[f"{name}_p84"] = q84[i]

    data["depth_ppm_map"] = fit.depth_ppm
    data["u1_map"] = fit.u1
    data["u2_map"] = fit.u2
    data["eclipse_depth_ppm_map"] = fit.eclipse_depth_ppm
    data["eclipse_depth_prior_mean_ppm"] = eclipse_depth_prior_mean_ppm
    data["f_min_ppm_map"] = fit.f_min_ppm
    data["c1_ppm_map"] = fit.c1_ppm
    data["peak_flux_ppm_map"] = fit.f_min_ppm + fit.c1_ppm
    data["t_peak_hr_map"] = fit.t_peak_hr
    data["tau_rise_hr_map"] = fit.tau_rise_hr
    data["tau_decay_hr_map"] = fit.tau_decay_hr
    data["jitter_ppm_map"] = fit.jitter_ppm

    data["a_over_rs_fixed"] = A_RS
    data["stellar_density_cgs"] = STELLAR_DENSITY_CGS
    data["period_days_fixed"] = P_ORB
    data["t0_transit_bjd_fixed"] = T0_TRANSIT
    data["eccentricity_fixed"] = ECC
    data["omega_deg_fixed"] = OMEGA_DEG
    data["inc_deg_fixed"] = INC_DEG
    data["t_periastron_ref_bjd"] = T_PERI_REF
    data["t_occultation_ref_bjd"] = T_OCC_REF
    data["phase_transit"] = PHASE_TRANSIT
    data["phase_periastron"] = PHASE_PERIASTRON
    data["phase_occultation"] = PHASE_OCCULTATION

    return pd.DataFrame([data])


def parse_args():
    parser = argparse.ArgumentParser(description="HAT-P-2b 4.5 um Phase 2 de Wit-style analysis.")
    parser.add_argument("--phase1-csv", type=Path, default=Path("output/phase1_hatp2b_45um_photometry.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--prefix", type=str, default="phase2_hatp2b_45um")
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--walkers", type=int, default=DEFAULT_N_WALKERS)
    parser.add_argument("--steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--burn", type=int, default=DEFAULT_N_BURN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-emcee", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.bin_width <= 0.0 or args.bin_width >= 1.0:
        raise ValueError("--bin-width must lie in (0, 1).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.prefix

    LOG.info("Loading Phase 1 product: %s", args.phase1_csv)
    df = load_phase1_photometry(args.phase1_csv)
    LOG.info("Loaded %d usable frames from %d AORs.", len(df), df["aor_id"].nunique())
    LOG.info(
        "Fixed orbital geometry: a/Rs=%.4f (from stellar density), P=%.7f d, e=%.5f, w=%.2f deg, i=%.2f deg.",
        A_RS,
        P_ORB,
        ECC,
        OMEGA_DEG,
        INC_DEG,
    )
    LOG.info(
        "Reference phases: transit=%.4f, periastron=%.4f, occultation=%.4f.",
        PHASE_TRANSIT,
        PHASE_PERIASTRON,
        PHASE_OCCULTATION,
    )

    df = apply_pixel_map_per_aor(df)
    df["phase"] = phase_fold_01(df["bjd_utc"].to_numpy(dtype=float))
    df = normalize_per_aor_baseline(df)

    binned = bin_phase_curve(
        df["phase"].to_numpy(dtype=float),
        df["flux_corr_final"].to_numpy(dtype=float),
        bin_width=args.bin_width,
    )

    if len(binned) < 30:
        raise RuntimeError(f"Only {len(binned)} populated phase bins; cannot fit reliably.")

    LOG.info("Created %d populated phase bins.", len(binned))

    if args.skip_emcee:
        fit = fixed_dewit_summary()
        flat_chain = np.array([fit_summary_to_theta(fit)], dtype=float)
        eclipse_depth_prior_mean_ppm = measure_eclipse_depth_ppm(binned)
        LOG.info("Skipping EMCEE; using de Wit reference parameter guesses.")
    else:
        fit, flat_chain, _, eclipse_depth_prior_mean_ppm = run_emcee_fit(
            binned=binned, n_walkers=args.walkers, n_steps=args.steps,
            n_burn=args.burn, seed=args.seed,
        )

    LOG.info(
        "MAP fit: depth=%.1f ppm, u1=%.3f, u2=%.3f, eclipse_depth=%.1f ppm, "
        "Fmin=%.1f ppm, c1=%.1f ppm (peak=%.1f ppm), tpeak=%.3f hr, "
        "trise=%.3f hr, tdecay=%.3f hr, jitter=%.1f ppm.",
        fit.depth_ppm, fit.u1, fit.u2, fit.eclipse_depth_ppm,
        fit.f_min_ppm, fit.c1_ppm, fit.f_min_ppm + fit.c1_ppm,
        fit.t_peak_hr, fit.tau_rise_hr, fit.tau_decay_hr, fit.jitter_ppm,
    )

    theta = fit_summary_to_theta(fit)

    df["model_flux"] = astrophysical_flux(df["bjd_utc"].to_numpy(dtype=float), theta)
    df["residual_flux"] = df["flux_corr_final"].to_numpy(dtype=float) - df["model_flux"].to_numpy(dtype=float)
    df["residual_ppm"] = df["residual_flux"] * 1.0e6

    binned_time = T0_TRANSIT + (binned["phase_center"].to_numpy(dtype=float) - PHASE_TRANSIT) * P_ORB
    binned["model_flux"] = astrophysical_flux(binned_time, theta)
    binned["residual_flux"] = binned["flux_median"].to_numpy(dtype=float) - binned["model_flux"].to_numpy(dtype=float)
    binned["residual_ppm"] = binned["residual_flux"] * 1.0e6

    lightcurve_path = prefix.with_name(prefix.name + "_corrected_lightcurve.csv")
    binned_path = prefix.with_name(prefix.name + "_phase_binned.csv")
    summary_path = prefix.with_name(prefix.name + "_posterior_summary.csv")
    figure_path = prefix.with_name(prefix.name + "_fig1_no_pulsations.png")

    df.to_csv(lightcurve_path, index=False)
    binned.to_csv(binned_path, index=False)
    posterior_summary_dataframe(flat_chain, fit, eclipse_depth_prior_mean_ppm).to_csv(summary_path, index=False)

    plot_dewit_fig1_style(binned=binned, fit=fit, output_png=figure_path)

    LOG.info("Saved corrected Phase 2 light curve: %s", lightcurve_path)
    LOG.info("Saved binned phase curve: %s", binned_path)
    LOG.info("Saved posterior summary: %s", summary_path)
    LOG.info("Saved Figure 1 reproduction: %s", figure_path)


if __name__ == "__main__":
    main()