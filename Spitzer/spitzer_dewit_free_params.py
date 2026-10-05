from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import batman
import emcee
import matplotlib.pyplot as plt
import numpy as np

try:
    import corner
except ImportError:
    corner = None
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

LOG = logging.getLogger("hatp2_phase2")

P_ORB = 5.6334675
T0_TRANSIT = 2455288.84969
ECC = 0.51023
OMEGA_DEG = 188.44
INC_DEG = 86.16
STELLAR_DENSITY_CGS = 0.434
G_CGS = 6.67430e-8

EXPOSURE_SECONDS = 0.4
EXPOSURE_DAYS = EXPOSURE_SECONDS / 86400.0
BATMAN_SUPERSAMPLE = 7
PHASE_WRAP_OFFSET = 0.05

DEFAULT_BIN_WIDTH = 0.00025
BIN_ERROR_FLOOR_PPM = 25.0

N_IP_NEIGHBORS = 50
MIN_IP_POINTS = 200
MIN_KERNEL_SIGMA = 0.20
IP_QUERY_CHUNK_SIZE = 5000
IP_LOCAL_PASSES = 2

MAX_EVENT_RAMP_PPM_PER_HR = 500.0
ECLIPSE_PRIOR_PPM = 971.0
ECLIPSE_PRIOR_SIGMA_PPM = 60.0

DEFAULT_N_WALKERS = 64
DEFAULT_N_STEPS = 6000
DEFAULT_N_BURN = 2500
DEFAULT_OUTER_ITERATIONS = 2
RANDOM_SEED = 24601

PARAM_NAMES = (
    "depth_ppm",
    "u1",
    "u2",
    "f_min_ppm",
    "c1_ppm",
    "t_peak_hr",
    "tau_rise_hr",
    "tau_decay_hr",
    "jitter_ppm",
)

PRIOR_BOUNDS = {
    "depth_ppm": (3500.0, 6500.0),
    "u1": (-0.30, 1.20),
    "u2": (-0.30, 1.20),
    "f_min_ppm": (0.0, 1000.0),
    "c1_ppm": (100.0, 1800.0),
    "t_peak_hr": (-2.0, 18.0),
    "tau_rise_hr": (1.0, 20.0),
    "tau_decay_hr": (2.0, 35.0),
    "jitter_ppm": (0.0, 3000.0),
}

THETA_DEWIT = np.array(
    [4941.0, 0.06, 0.23, 322.0, 856.0, 5.40, 5.5, 10.3, 100.0],
    dtype=float,
)

WALKER_SPREAD = np.array(
    [50.0, 0.025, 0.025, 30.0, 40.0, 0.35, 0.50, 0.80, 20.0],
    dtype=float,
)


# General helpers
def robust_location(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def mad_std(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad)


def as_bool_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "t", "1", "yes", "y"])
        .to_numpy(dtype=bool)
    )


def derive_a_over_rs(stellar_density_cgs: float, period_days: float) -> float:
    period_seconds = period_days * 86400.0
    value = stellar_density_cgs * G_CGS * period_seconds**2 / (3.0 * np.pi)
    return float(value ** (1.0 / 3.0))


A_RS = derive_a_over_rs(STELLAR_DENSITY_CGS, P_ORB)


def phase_fold_01(
    bjd: np.ndarray,
    period_days: float = P_ORB,
    t0_bjd: float = T0_TRANSIT,
    wrap_offset: float = PHASE_WRAP_OFFSET,
) -> np.ndarray:
    raw = ((np.asarray(bjd, dtype=float) - t0_bjd) / period_days) % 1.0
    return (raw + wrap_offset) % 1.0


def centered_phase(phase_01: np.ndarray, center_01: float) -> np.ndarray:
    phase_01 = np.asarray(phase_01, dtype=float)
    return ((phase_01 - center_01 + 0.5) % 1.0) - 0.5


def circular_phase_distance(phase_01: np.ndarray, center_01: float) -> np.ndarray:
    return np.abs(centered_phase(phase_01, center_01))


def true_to_mean_anomaly(true_anomaly: float, eccentricity: float) -> float:
    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly),
    )
    return float((eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) % (2.0 * np.pi))


def transit_to_periastron_time(
    t_transit_bjd: float,
    period_days: float,
    eccentricity: float,
    omega_deg: float,
) -> float:
    f_transit = 0.5 * np.pi - np.deg2rad(omega_deg)
    m_transit = true_to_mean_anomaly(f_transit, eccentricity)
    return float(t_transit_bjd - period_days * m_transit / (2.0 * np.pi))


def transit_to_occultation_time(
    t_transit_bjd: float,
    period_days: float,
    eccentricity: float,
    omega_deg: float,
) -> float:
    omega = np.deg2rad(omega_deg)
    m_transit = true_to_mean_anomaly(0.5 * np.pi - omega, eccentricity)
    m_occultation = true_to_mean_anomaly(1.5 * np.pi - omega, eccentricity)
    delta_m = (m_occultation - m_transit) % (2.0 * np.pi)
    return float(t_transit_bjd + period_days * delta_m / (2.0 * np.pi))


T_PERI_REF = transit_to_periastron_time(T0_TRANSIT, P_ORB, ECC, OMEGA_DEG)
T_OCC_REF = transit_to_occultation_time(T0_TRANSIT, P_ORB, ECC, OMEGA_DEG)
PHASE_TRANSIT = float(phase_fold_01(np.array([T0_TRANSIT]))[0])
PHASE_PERIASTRON = float(phase_fold_01(np.array([T_PERI_REF]))[0])
PHASE_OCCULTATION = float(phase_fold_01(np.array([T_OCC_REF]))[0])


def nearest_periodic_epoch(
    reference_event_bjd: float,
    query_times_bjd: np.ndarray,
    period_days: float = P_ORB,
) -> np.ndarray:
    query_times_bjd = np.asarray(query_times_bjd, dtype=float)
    cycle = np.rint((query_times_bjd - reference_event_bjd) / period_days)
    return reference_event_bjd + cycle * period_days


def hours_since_periastron(time_bjd: np.ndarray) -> np.ndarray:
    time_bjd = np.asarray(time_bjd, dtype=float)
    nearest_peri = nearest_periodic_epoch(T_PERI_REF, time_bjd)
    return (time_bjd - nearest_peri) * 24.0

def make_geometry_params(rp_rs: float, u1: float, u2: float) -> batman.TransitParams:
    params = batman.TransitParams()
    params.t0 = T0_TRANSIT
    params.per = P_ORB
    params.rp = float(rp_rs)
    params.a = A_RS
    params.inc = INC_DEG
    params.ecc = ECC
    params.w = OMEGA_DEG
    params.t_secondary = T_OCC_REF
    params.limb_dark = "quadratic"
    params.u = [float(u1), float(u2)]
    params.fp = 0.0
    return params


def compute_transit_shape(
    time_bjd: np.ndarray,
    rp_rs: float,
    u1: float,
    u2: float,
) -> np.ndarray:
    time_bjd = np.asarray(time_bjd, dtype=float)
    params = make_geometry_params(rp_rs, u1, u2)
    model = batman.TransitModel(
        params,
        time_bjd,
        transittype="primary",
        supersample_factor=BATMAN_SUPERSAMPLE,
        exp_time=EXPOSURE_DAYS,
    )
    return model.light_curve(params)


def compute_eclipse_window(time_bjd: np.ndarray, rp_rs: float) -> np.ndarray:
    """Return the BATMAN occulted fraction: 0 outside, 1 in full eclipse."""
    time_bjd = np.asarray(time_bjd, dtype=float)
    params = make_geometry_params(rp_rs, 0.0, 0.0)
    params.fp = 1.0
    params.limb_dark = "uniform"
    params.u = []

    model = batman.TransitModel(
        params,
        time_bjd,
        transittype="secondary",
        supersample_factor=BATMAN_SUPERSAMPLE,
        exp_time=EXPOSURE_DAYS,
    )
    secondary_flux = model.light_curve(params)
    window = ((1.0 + params.fp) - secondary_flux) / params.fp

    tolerance = 1.0e-12
    window[window <= tolerance] = 0.0
    window[window >= 1.0 - tolerance] = 1.0
    return np.clip(window, 0.0, 1.0)


def asymmetric_lorentzian_ppm(
    hours_from_periastron: np.ndarray,
    f_min_ppm: float,
    c1_ppm: float,
    t_peak_hr: float,
    tau_rise_hr: float,
    tau_decay_hr: float,
) -> np.ndarray:
    t = np.asarray(hours_from_periastron, dtype=float)
    tau_rise_hr = max(float(tau_rise_hr), 1.0e-6)
    tau_decay_hr = max(float(tau_decay_hr), 1.0e-6)
    u = np.where(
        t < t_peak_hr,
        (t - t_peak_hr) / tau_rise_hr,
        (t - t_peak_hr) / tau_decay_hr,
    )
    return float(f_min_ppm) + float(c1_ppm) / (1.0 + u * u)


def unpack_theta(theta: np.ndarray) -> tuple[float, ...]:
    if len(theta) != len(PARAM_NAMES):
        raise ValueError(f"Expected {len(PARAM_NAMES)} parameters, received {len(theta)}")
    return tuple(float(value) for value in theta)


class AstrophysicalEvaluator:
    def __init__(self, time_bjd: np.ndarray):
        self.time_bjd = np.asarray(time_bjd, dtype=float)
        self.hours_from_peri = hours_since_periastron(self.time_bjd)

        rp_ref = np.sqrt(4941.0e-6)
        self.primary_params = make_geometry_params(rp_ref, 0.06, 0.23)
        self.primary_model = batman.TransitModel(
            self.primary_params,
            self.time_bjd,
            transittype="primary",
            supersample_factor=BATMAN_SUPERSAMPLE,
            exp_time=EXPOSURE_DAYS,
        )

        self.secondary_params = make_geometry_params(rp_ref, 0.0, 0.0)
        self.secondary_params.fp = 1.0
        self.secondary_params.limb_dark = "uniform"
        self.secondary_params.u = []
        self.secondary_model = batman.TransitModel(
            self.secondary_params,
            self.time_bjd,
            transittype="secondary",
            supersample_factor=BATMAN_SUPERSAMPLE,
            exp_time=EXPOSURE_DAYS,
        )

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        (
            depth_ppm,
            u1,
            u2,
            f_min_ppm,
            c1_ppm,
            t_peak_hr,
            tau_rise_hr,
            tau_decay_hr,
            _jitter_ppm,
        ) = unpack_theta(theta)

        rp_rs = np.sqrt(max(depth_ppm, 1.0) * 1.0e-6)

        self.primary_params.rp = rp_rs
        self.primary_params.u = [u1, u2]
        stellar_flux = self.primary_model.light_curve(self.primary_params)

        self.secondary_params.rp = rp_rs
        secondary_flux = self.secondary_model.light_curve(self.secondary_params)
        eclipse_window = 2.0 - secondary_flux
        eclipse_window[eclipse_window <= 1.0e-12] = 0.0
        eclipse_window[eclipse_window >= 1.0 - 1.0e-12] = 1.0
        eclipse_window = np.clip(eclipse_window, 0.0, 1.0)

        planet_ppm = asymmetric_lorentzian_ppm(
            self.hours_from_peri,
            f_min_ppm,
            c1_ppm,
            t_peak_hr,
            tau_rise_hr,
            tau_decay_hr,
        )

        return stellar_flux + planet_ppm * 1.0e-6 * (1.0 - eclipse_window)


def astrophysical_flux(time_bjd: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return AstrophysicalEvaluator(time_bjd).evaluate(theta)


def eclipse_depth_from_theta(theta: np.ndarray) -> float:
    """Planet/star flux at occultation center, in ppm."""
    values = unpack_theta(theta)
    t_occ_hr = float(hours_since_periastron(np.array([T_OCC_REF]))[0])
    return float(
        asymmetric_lorentzian_ppm(
            np.array([t_occ_hr]),
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
        )[0]
    )

def standardized_coordinates(
    x_cent: np.ndarray,
    y_cent: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    arrays = [np.asarray(x_cent, float), np.asarray(y_cent, float), np.asarray(beta, float)]
    scales = []
    for index, values in enumerate(arrays):
        floor = 0.002 if index < 2 else 0.02
        scale = mad_std(values)
        scales.append(max(scale if np.isfinite(scale) else floor, floor))
    return np.column_stack(
        [(values - np.median(values)) / scale for values, scale in zip(arrays, scales)]
    )


def leave_one_out_pixel_map(
    x_cent: np.ndarray,
    y_cent: np.ndarray,
    beta: np.ndarray,
    flattened_flux: np.ndarray,
    n_neighbors: int = N_IP_NEIGHBORS,
) -> np.ndarray:
    x_cent = np.asarray(x_cent, dtype=float)
    y_cent = np.asarray(y_cent, dtype=float)
    beta = np.asarray(beta, dtype=float)
    flattened_flux = np.asarray(flattened_flux, dtype=float)
    n_points = flattened_flux.size

    if n_points < MIN_IP_POINTS:
        return np.ones(n_points, dtype=float)

    coordinates = standardized_coordinates(x_cent, y_cent, beta)
    tree = cKDTree(coordinates)
    query_k = min(n_points, n_neighbors + 1)
    sensitivity = np.ones(n_points, dtype=float)

    for start in range(0, n_points, IP_QUERY_CHUNK_SIZE):
        stop = min(start + IP_QUERY_CHUNK_SIZE, n_points)
        _, candidates = tree.query(
            coordinates[start:stop],
            k=query_k,
            workers=-1,
        )
        if candidates.ndim == 1:
            candidates = candidates[:, None]

        row_ids = np.arange(start, stop)[:, None]
        is_self = candidates == row_ids
      
        reorder = np.argsort(is_self, axis=1, kind="stable")
        candidates = np.take_along_axis(candidates, reorder, axis=1)[:, :n_neighbors]
        valid_candidate = candidates != row_ids
        safe_candidates = np.where(valid_candidate, candidates, 0)

        delta = coordinates[safe_candidates] - coordinates[start:stop, None, :]
        delta[~valid_candidate] = np.nan
        sigma = np.nanstd(delta, axis=1, ddof=1)
        sigma = np.where(np.isfinite(sigma), np.maximum(sigma, MIN_KERNEL_SIGMA), 1.0)

        exponent = -0.5 * np.nansum((delta / sigma[:, None, :]) ** 2, axis=2)
        weights = np.exp(np.clip(exponent, -700.0, 0.0))
        weights[~valid_candidate] = 0.0

        neighbor_flux = flattened_flux[safe_candidates]
        valid_flux = np.isfinite(neighbor_flux) & (neighbor_flux > 0.0)
        weights[~valid_flux] = 0.0
        weight_sum = np.sum(weights, axis=1)
        estimate = np.divide(
            np.sum(weights * np.where(valid_flux, neighbor_flux, 0.0), axis=1),
            weight_sum,
            out=np.ones(stop - start, dtype=float),
            where=weight_sum > 0.0,
        )
        sensitivity[start:stop] = estimate

    valid = np.isfinite(sensitivity) & (sensitivity > 0.0)
    if not np.any(valid):
        return np.ones(n_points, dtype=float)
    sensitivity[~valid] = np.median(sensitivity[valid])
    sensitivity /= np.median(sensitivity[valid])
    return sensitivity


def robust_linear_baseline(
    time_bjd: np.ndarray,
    ratio: np.ndarray,
    fit_mask: np.ndarray,
    allow_slope: bool,
) -> tuple[np.ndarray, float, float]:
    time_bjd = np.asarray(time_bjd, float)
    ratio = np.asarray(ratio, float)
    fit_mask = np.asarray(fit_mask, bool)
    t_center = robust_location(time_bjd[fit_mask])
    if not np.isfinite(t_center):
        t_center = robust_location(time_bjd)
    x = (time_bjd - t_center) * 24.0

    good = fit_mask & np.isfinite(x) & np.isfinite(ratio) & (ratio > 0.0)
    degree = 1 if allow_slope and good.sum() >= 40 else 0

    if good.sum() < 10:
        intercept = robust_location(ratio)
        intercept = intercept if np.isfinite(intercept) and intercept > 0.0 else 1.0
        return np.full_like(ratio, intercept), intercept, 0.0

    active = good.copy()
    coefficients = np.array([robust_location(ratio[good])])
    for _ in range(5):
        coefficients = np.polyfit(x[active], ratio[active], degree)
        prediction = np.polyval(coefficients, x)
        residual = ratio - prediction
        scatter = mad_std(residual[active])
        if not np.isfinite(scatter) or scatter <= 0.0:
            break
        new_active = good & (np.abs(residual) <= 4.0 * scatter)
        if new_active.sum() == active.sum():
            break
        if new_active.sum() < degree + 10:
            break
        active = new_active

    if degree == 0:
        intercept = float(coefficients[-1])
        slope = 0.0
    else:
        slope = float(coefficients[0])
        intercept = float(coefficients[1])
        slope_limit = MAX_EVENT_RAMP_PPM_PER_HR * 1.0e-6 * max(intercept, 1.0e-6)
        slope = float(np.clip(slope, -slope_limit, slope_limit))

    baseline = intercept + slope * x
    if not np.isfinite(intercept) or intercept <= 0.0 or np.any(baseline <= 0.0):
        intercept = robust_location(ratio[good])
        intercept = intercept if np.isfinite(intercept) and intercept > 0.0 else 1.0
        slope = 0.0
        baseline = np.full_like(ratio, intercept)

    return baseline, intercept, slope


def classify_visit(label: str) -> str:
    label = str(label).strip().lower()
    if "occ" in label or "eclipse" in label:
        return "occultation"
    if "trans" in label:
        return "transit"
    if "phase" in label:
        return "phase"
    return "unknown"


def correct_photometry(df: pd.DataFrame, theta_reference: np.ndarray) -> pd.DataFrame:
    output = df.copy()
    output["ip_sensitivity"] = 1.0
    output["aor_baseline"] = 1.0
    output["aor_ramp_ppm_hr"] = 0.0
    output["flux_corr_ipix"] = np.nan
    output["flux_corr_final"] = np.nan

    rp_ref = np.sqrt(theta_reference[0] * 1.0e-6)

    for aor_id, group in output.groupby("aor_id", sort=False):
        indices = group.index.to_numpy()
        time_bjd = group["bjd_utc"].to_numpy(float)
        raw_flux = group["flux_norm_global"].to_numpy(float)
        x_cent = group["x_cent"].to_numpy(float)
        y_cent = group["y_cent"].to_numpy(float)
        beta = group["beta"].to_numpy(float)
        label = classify_visit(group["visit_label"].iloc[0])

        reference = astrophysical_flux(time_bjd, theta_reference)
        sensitivity = np.ones(len(group), dtype=float)
        baseline = np.ones(len(group), dtype=float)
        intercept = 1.0
        slope = 0.0

        for _ in range(IP_LOCAL_PASSES):
            flattened = raw_flux / (reference * baseline)
            sensitivity = leave_one_out_pixel_map(
                x_cent,
                y_cent,
                beta,
                flattened,
            )
            corrected = raw_flux / sensitivity
            ratio = corrected / reference

            if label == "occultation":
                event_shape = compute_eclipse_window(time_bjd, rp_ref)
                baseline_mask = event_shape <= 1.0e-10
                allow_slope = True
            elif label == "transit":
                event_shape = compute_transit_shape(time_bjd, rp_ref, theta_reference[1], theta_reference[2])
                baseline_mask = np.abs(event_shape - 1.0) <= 1.0e-10
                allow_slope = True
            else:
                baseline_mask = np.ones(len(group), dtype=bool)
                allow_slope = False

            baseline, intercept, slope = robust_linear_baseline(
                time_bjd,
                ratio,
                baseline_mask,
                allow_slope,
            )

        corrected_ip = raw_flux / sensitivity
        corrected_final = corrected_ip / baseline

        output.loc[indices, "ip_sensitivity"] = sensitivity
        output.loc[indices, "aor_baseline"] = baseline
        output.loc[indices, "aor_ramp_ppm_hr"] = slope / max(intercept, 1.0e-12) * 1.0e6
        output.loc[indices, "flux_corr_ipix"] = corrected_ip
        output.loc[indices, "flux_corr_final"] = corrected_final

        LOG.info(
            "AOR %s (%s): N=%d, scale=%.8f, ramp=%+.1f ppm/hr",
            aor_id,
            label,
            len(group),
            intercept,
            slope / max(intercept, 1.0e-12) * 1.0e6,
        )

    return output

def bin_phase_curve(
    phase_01: np.ndarray,
    flux: np.ndarray,
    aor_id: np.ndarray,
    bin_width: float = DEFAULT_BIN_WIDTH,
) -> pd.DataFrame:
    phase_01 = np.asarray(phase_01, float)
    flux = np.asarray(flux, float)
    aor_id = np.asarray(aor_id)
    valid = (
        np.isfinite(phase_01)
        & np.isfinite(flux)
        & (phase_01 >= 0.0)
        & (phase_01 < 1.0)
        & (flux > 0.0)
    )
    phase_01 = phase_01[valid]
    flux = flux[valid]
    aor_id = aor_id[valid]

    n_bins = int(np.ceil(1.0 / bin_width))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(phase_01, edges) - 1, 0, n_bins - 1)
    table = pd.DataFrame({"phase": phase_01, "flux": flux, "aor_id": aor_id, "bin_id": bin_id})

    rows = []
    for this_bin, group in table.groupby("bin_id", sort=True):
        values = group["flux"].to_numpy(float)
        n_points = values.size
        scatter = mad_std(values)
        if not np.isfinite(scatter) or scatter <= 0.0:
            scatter = float(np.std(values, ddof=1)) if n_points > 1 else np.nan
        uncertainty_ppm = (
            scatter * 1.0e6 / np.sqrt(n_points)
            if np.isfinite(scatter) and n_points > 1
            else BIN_ERROR_FLOOR_PPM
        )
        uncertainty_ppm = max(uncertainty_ppm, BIN_ERROR_FLOOR_PPM)
        rows.append(
            {
                "bin_id": int(this_bin),
                "phase_center": 0.5 * (edges[this_bin] + edges[this_bin + 1]),
                "phase_median": float(np.median(group["phase"])),
                "flux_median": float(np.median(values)),
                "flux_mean": float(np.mean(values)),
                "flux_err": uncertainty_ppm * 1.0e-6,
                "flux_err_ppm": uncertainty_ppm,
                "n_points": int(n_points),
                "n_aors": int(group["aor_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)

def limb_darkening_is_physical(u1: float, u2: float) -> bool:
    return (u1 + u2 < 1.0) and (u1 > 0.0) and (u1 + 2.0 * u2 > 0.0)


def gaussian_log_prior(value: float, mean: float, sigma: float) -> float:
    return -0.5 * ((value - mean) / sigma) ** 2


def log_prior(theta: np.ndarray, use_reference_priors: bool = True) -> float:
    for name, value in zip(PARAM_NAMES, theta):
        lower, upper = PRIOR_BOUNDS[name]
        if not lower <= value <= upper:
            return -np.inf

    if not limb_darkening_is_physical(theta[1], theta[2]):
        return -np.inf

    if not use_reference_priors:
        return 0.0

    lp = 0.0
    lp += gaussian_log_prior(theta[0], 4941.0, 150.0)
    lp += gaussian_log_prior(theta[1], 0.06, 0.20)
    lp += gaussian_log_prior(theta[2], 0.23, 0.20)
    lp += gaussian_log_prior(theta[3], 322.0, 180.0)
    lp += gaussian_log_prior(theta[3] + theta[4], 1178.0, 140.0)
    lp += gaussian_log_prior(theta[5], 5.40, 2.0)
    lp += gaussian_log_prior(theta[6], 5.5, 2.5)
    lp += gaussian_log_prior(theta[7], 10.3, 4.0)
    lp += gaussian_log_prior(eclipse_depth_from_theta(theta), ECLIPSE_PRIOR_PPM, ECLIPSE_PRIOR_SIGMA_PPM)
    return float(lp)


def build_fit_arrays(binned: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = binned["phase_center"].to_numpy(float)
    time_bjd = T0_TRANSIT + (phase - PHASE_TRANSIT) * P_ORB
    flux = binned["flux_median"].to_numpy(float)
    error = binned["flux_err"].to_numpy(float)
    valid = np.isfinite(time_bjd) & np.isfinite(flux) & np.isfinite(error) & (error > 0.0)
    return time_bjd[valid], flux[valid], error[valid]


def make_log_probability(
    evaluator: AstrophysicalEvaluator,
    flux: np.ndarray,
    error: np.ndarray,
    use_reference_priors: bool,
):
    def log_probability(theta: np.ndarray) -> float:
        lp = log_prior(theta, use_reference_priors)
        if not np.isfinite(lp):
            return -np.inf
        model = evaluator.evaluate(theta)
        jitter = theta[-1] * 1.0e-6
        variance = error * error + jitter * jitter
        residual = flux - model
        ll = -0.5 * np.sum(residual * residual / variance + np.log(2.0 * np.pi * variance))
        return float(lp + ll) if np.isfinite(ll) else -np.inf

    return log_probability


def fit_map(
    binned: pd.DataFrame,
    theta_start: np.ndarray,
    use_reference_priors: bool,
) -> np.ndarray:
    time_bjd, flux, error = build_fit_arrays(binned)
    evaluator = AstrophysicalEvaluator(time_bjd)
    log_probability = make_log_probability(evaluator, flux, error, use_reference_priors)

    bounds = [PRIOR_BOUNDS[name] for name in PARAM_NAMES]

    def objective(theta: np.ndarray) -> float:
        value = log_probability(theta)
        return -value if np.isfinite(value) else 1.0e100

    result = minimize(
        objective,
        np.asarray(theta_start, float),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 3000, "xtol": 1.0e-6, "ftol": 1.0e-7},
    )
    candidate = result.x if np.isfinite(log_probability(result.x)) else np.asarray(theta_start, float)
    LOG.info("MAP optimization success=%s; log posterior=%.2f", result.success, log_probability(candidate))
    log_parameters(candidate, prefix="MAP")
    return candidate


@dataclass
class FitResult:
    theta_map: np.ndarray
    sampler: emcee.EnsembleSampler | None = None
    n_burn: int = 0


def initialize_walkers(
    theta_start: np.ndarray,
    n_walkers: int,
    log_probability,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = np.empty((n_walkers, len(theta_start)), dtype=float)
    for walker in range(n_walkers):
        accepted = False
        for _ in range(1000):
            candidate = theta_start + WALKER_SPREAD * rng.normal(size=len(theta_start))
            if np.isfinite(log_probability(candidate)):
                positions[walker] = candidate
                accepted = True
                break
        if not accepted:
            raise RuntimeError("Could not initialize all EMCEE walkers inside the prior")
    return positions


def run_emcee_fit(
    binned: pd.DataFrame,
    theta_start: np.ndarray,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    seed: int,
    use_reference_priors: bool,
) -> FitResult:
    if n_burn >= n_steps:
        raise ValueError("--burn must be smaller than --steps")
    if n_walkers < 2 * len(PARAM_NAMES):
        raise ValueError(f"--walkers must be at least {2 * len(PARAM_NAMES)}")

    time_bjd, flux, error = build_fit_arrays(binned)
    evaluator = AstrophysicalEvaluator(time_bjd)
    log_probability = make_log_probability(evaluator, flux, error, use_reference_priors)
    theta_map_start = fit_map(binned, theta_start, use_reference_priors)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    initial = initialize_walkers(theta_map_start, n_walkers, log_probability, rng)

    sampler = emcee.EnsembleSampler(n_walkers, len(PARAM_NAMES), log_probability)
    LOG.info("Running EMCEE: %d walkers x %d steps; burn=%d", n_walkers, n_steps, n_burn)
    sampler.run_mcmc(initial, n_steps, progress=True)

    flat_chain = sampler.get_chain(discard=n_burn, flat=True)
    flat_log_probability = sampler.get_log_prob(discard=n_burn, flat=True)
    best = int(np.nanargmax(flat_log_probability))
    theta_map = flat_chain[best].copy()

    LOG.info("Mean post-burn acceptance fraction: %.3f", np.mean(sampler.acceptance_fraction))
    log_parameters(theta_map, prefix="EMCEE MAP")
    return FitResult(theta_map=theta_map, sampler=sampler, n_burn=n_burn)


def log_parameters(theta: np.ndarray, prefix: str = "fit") -> None:
    for name, value in zip(PARAM_NAMES, theta):
        LOG.info("%s %-16s = %.6g", prefix, name, value)
    LOG.info("%s derived eclipse depth = %.2f ppm", prefix, eclipse_depth_from_theta(theta))

def split_rhat(post_chain: np.ndarray) -> np.ndarray:
    post_chain = np.asarray(post_chain, dtype=float)
    n_steps, _n_walkers, n_parameters = post_chain.shape
    half = n_steps // 2
    if half < 4:
        return np.full(n_parameters, np.nan)

    split = np.concatenate((post_chain[:half], post_chain[-half:]), axis=1)
    n = split.shape[0]
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    between = n * np.var(np.mean(split, axis=0), axis=0, ddof=1)
    variance = ((n - 1.0) / n) * within + between / n
    return np.sqrt(variance / within)


def mcmc_statistics(result: FitResult) -> dict[str, np.ndarray | float]:
    if result.sampler is None:
        raise ValueError("MCMC diagnostics require an EMCEE run")

    sampler = result.sampler
    post_chain = sampler.get_chain(discard=result.n_burn, flat=False)
    flat_chain = sampler.get_chain(discard=result.n_burn, flat=True)
    n_post_steps, n_walkers, _ = post_chain.shape

    try:
        tau = np.asarray(sampler.get_autocorr_time(discard=result.n_burn, tol=0), dtype=float)
    except Exception as exc:
        LOG.warning("Could not estimate autocorrelation time: %s", exc)
        tau = np.full(len(PARAM_NAMES), np.nan)

    ess = np.divide(
        n_post_steps * n_walkers,
        tau,
        out=np.full(len(PARAM_NAMES), np.nan),
        where=np.isfinite(tau) & (tau > 0.0),
    )
    steps_per_tau = np.divide(
        n_post_steps,
        tau,
        out=np.full(len(PARAM_NAMES), np.nan),
        where=np.isfinite(tau) & (tau > 0.0),
    )
    rhat = split_rhat(post_chain)
    q16, q50, q84 = np.percentile(flat_chain, [16.0, 50.0, 84.0], axis=0)
    acceptance = np.asarray(sampler.acceptance_fraction, dtype=float)

    LOG.info("Posterior diagnostics (not used to alter the model or MAP solution):")
    LOG.info("  mean acceptance fraction = %.3f", np.mean(acceptance))
    LOG.info("  %-16s %12s %12s %12s %9s %10s %10s",
             "parameter", "median", "-1sigma", "+1sigma", "Rhat", "ESS", "steps/tau")
    for index, name in enumerate(PARAM_NAMES):
        LOG.info(
            "  %-16s %12.5g %12.4g %12.4g %9.4f %10.0f %10.1f",
            name, q50[index], q50[index] - q16[index], q84[index] - q50[index],
            rhat[index], ess[index], steps_per_tau[index],
        )

    if np.any(np.isfinite(rhat) & (rhat > 1.01)):
        LOG.warning("At least one split-Rhat is > 1.01; inspect the trace plot.")
    if np.any(np.isfinite(ess) & (ess < 400.0)):
        LOG.warning("At least one effective sample size is < 400.")
    if np.any(np.isfinite(steps_per_tau) & (steps_per_tau < 50.0)):
        LOG.warning("At least one post-burn chain is shorter than 50 autocorrelation times.")

    return {
        "post_chain": post_chain,
        "flat_chain": flat_chain,
        "tau": tau,
        "ess": ess,
        "steps_per_tau": steps_per_tau,
        "rhat": rhat,
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "acceptance_mean": float(np.mean(acceptance)),
    }


def plot_mcmc_traces_and_convergence(
    result: FitResult,
    statistics: dict[str, np.ndarray | float],
    output_png: Path,
) -> None:
    if result.sampler is None:
        return

    chain = result.sampler.get_chain()
    tau = np.asarray(statistics["tau"])
    ess = np.asarray(statistics["ess"])
    rhat = np.asarray(statistics["rhat"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    fig, axes = plt.subplots(len(PARAM_NAMES), 1, figsize=(11.0, 13.0), sharex=True)
    for index, (axis, name) in enumerate(zip(axes, PARAM_NAMES)):
        axis.plot(chain[:, :, index], color="black", alpha=0.10, lw=0.35, rasterized=True)
        axis.axvline(result.n_burn, color="#00a020", ls="--", lw=1.0)
        axis.set_ylabel(name, rotation=0, ha="right", va="center")
        diagnostic_text = (
            rf"$\hat{{R}}={rhat[index]:.3f}$   "
            rf"$\tau={tau[index]:.1f}$   ESS={ess[index]:.0f}"
        )
        axis.text(0.995, 0.82, diagnostic_text, transform=axis.transAxes,
                  ha="right", va="top", fontsize=8,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75})
        axis.grid(alpha=0.10, lw=0.4)
    axes[-1].set_xlabel("MCMC step")
    fig.suptitle(
        f"EMCEE traces and convergence  |  mean acceptance = {statistics['acceptance_mean']:.3f}\n"
        "green dashed line: burn-in boundary",
        y=0.998,
    )
    fig.tight_layout(rect=(0.04, 0.02, 1.0, 0.975))
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", output_png)


def plot_corner_posterior(
    result: FitResult,
    statistics: dict[str, np.ndarray | float],
    output_png: Path,
    seed: int = RANDOM_SEED,
) -> None:
    if corner is None:
        LOG.warning("Package 'corner' is not installed; skipping corner plot (pip install corner).")
        return

    flat_chain = np.asarray(statistics["flat_chain"])
    max_corner_samples = 60000
    if flat_chain.shape[0] > max_corner_samples:
        rng = np.random.default_rng(seed)
        use = np.sort(rng.choice(flat_chain.shape[0], max_corner_samples, replace=False))
        plot_chain = flat_chain[use]
    else:
        plot_chain = flat_chain

    labels = [
        r"$D_{\rm tr}$ [ppm]", r"$u_1$", r"$u_2$",
        r"$F_{\min}$ [ppm]", r"$c_1$ [ppm]", r"$t_{\rm peak}$ [hr]",
        r"$\tau_{\rm rise}$ [hr]", r"$\tau_{\rm decay}$ [hr]",
        r"$\sigma_{\rm jit}$ [ppm]",
    ]
    figure = corner.corner(
        plot_chain,
        labels=labels,
        truths=result.theta_map,
        truth_color="#2060c0",
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 10},
        color="#07852f",
        smooth=0.75,
        smooth1d=0.75,
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.393, 0.865),
        max_n_ticks=4,
    )
    figure.suptitle("HAT-P-2b 4.5 µm posterior", fontsize=15, y=1.005)
    figure.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(figure)
    LOG.info("Saved %s", output_png)


def validate_eclipse_window() -> None:
    phase = np.linspace(0.0, 1.0, 20001, endpoint=False)
    time_bjd = T0_TRANSIT + (phase - PHASE_TRANSIT) * P_ORB
    window = compute_eclipse_window(time_bjd, np.sqrt(4941.0e-6))
    far_from_occultation = circular_phase_distance(phase, PHASE_OCCULTATION) > 0.04
    near_transit = circular_phase_distance(phase, PHASE_TRANSIT) < 0.04
    leakage_far = float(np.max(window[far_from_occultation]))
    leakage_transit = float(np.max(window[near_transit]))
    if leakage_far > 1.0e-10 or leakage_transit > 1.0e-10:
        raise RuntimeError(
            f"Eclipse-window leakage detected: far={leakage_far:g}, transit={leakage_transit:g}"
        )
    LOG.info("Eclipse-window validation passed; no out-of-event leakage")


def plot_dewit_fig1_style(
    binned: pd.DataFrame,
    theta: np.ndarray,
    output_png: Path,
) -> None:
    phase_grid = np.linspace(0.0, 1.0, 24001, endpoint=False)
    time_grid = T0_TRANSIT + (phase_grid - PHASE_TRANSIT) * P_ORB
    model_grid = astrophysical_flux(time_grid, theta)

    phase_bin = binned["phase_center"].to_numpy(float)
    flux_bin = binned["flux_median"].to_numpy(float)
    time_bin = T0_TRANSIT + (phase_bin - PHASE_TRANSIT) * P_ORB
    model_bin = astrophysical_flux(time_bin, theta)
    residual_ppm = (flux_bin - model_bin) * 1.0e6

    transit_phase = centered_phase(phase_bin, PHASE_TRANSIT)
    transit_grid = centered_phase(phase_grid, PHASE_TRANSIT)
    occultation_phase = centered_phase(phase_bin, PHASE_OCCULTATION)
    occultation_grid = centered_phase(phase_grid, PHASE_OCCULTATION)
    transit_window = 0.030
    occultation_window = 0.030

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
    fig = plt.figure(figsize=(9.4, 9.2), dpi=180)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.08, 1.0, 0.75], hspace=0.42, wspace=0.30)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, :])

    ax_a.plot(phase_bin, flux_bin, "o", ms=2.4, color="black", alpha=0.88, rasterized=True)
    ax_a.plot(phase_grid, model_grid, color="#00b300", lw=1.7, zorder=4)
    ax_a.axhline(1.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_a.set(xlim=(0.0, 1.0), xlabel="Orbital phase", ylabel=r"$F/F_\star$")
    ax_a.text(0.985, 0.94, "A", transform=ax_a.transAxes, ha="right", va="top", fontweight="bold")

    transit_points = np.abs(transit_phase) <= transit_window
    transit_model = np.abs(transit_grid) <= transit_window
    order = np.argsort(transit_grid[transit_model])
    ax_b.plot(
        transit_phase[transit_points],
        (flux_bin[transit_points] - 1.0) * 1.0e6,
        "o",
        ms=2.5,
        color="black",
        alpha=0.88,
        rasterized=True,
    )
    ax_b.plot(
        transit_grid[transit_model][order],
        (model_grid[transit_model][order] - 1.0) * 1.0e6,
        color="#00b300",
        lw=1.7,
    )
    ax_b.set(
        xlim=(-transit_window, transit_window),
        xlabel="Phase centered on transit",
        ylabel=r"$(F/F_\star-1)$ [ppm]",
    )
    ax_b.text(0.95, 0.92, "B", transform=ax_b.transAxes, ha="right", va="top", fontweight="bold")

    occultation_points = np.abs(occultation_phase) <= occultation_window
    occultation_model = np.abs(occultation_grid) <= occultation_window
    order = np.argsort(occultation_grid[occultation_model])
    ax_c.plot(
        occultation_phase[occultation_points],
        (flux_bin[occultation_points] - 1.0) * 1.0e6,
        "o",
        ms=2.5,
        color="black",
        alpha=0.88,
        rasterized=True,
    )
    ax_c.plot(
        occultation_grid[occultation_model][order],
        (model_grid[occultation_model][order] - 1.0) * 1.0e6,
        color="#00b300",
        lw=1.7,
    )
    ax_c.axhline(0.0, color="0.55", ls="--", lw=0.8, zorder=0)
    ax_c.set(
        xlim=(-occultation_window, occultation_window),
        xlabel="Phase centered on occultation",
        ylabel=r"$(F/F_\star-1)$ [ppm]",
    )
    ax_c.text(0.95, 0.92, "C", transform=ax_c.transAxes, ha="right", va="top", fontweight="bold")

    ax_d.axhline(0.0, color="#00b300", lw=1.4)
    ax_d.plot(phase_bin, residual_ppm, "o", ms=2.2, color="black", alpha=0.85, rasterized=True)
    ax_d.set(xlim=(0.0, 1.0), xlabel="Orbital phase", ylabel="Data - model [ppm]")
    ax_d.text(0.985, 0.90, "D", transform=ax_d.transAxes, ha="right", va="top", fontweight="bold")

    for axis in (ax_a, ax_b, ax_c, ax_d):
        axis.grid(alpha=0.12, lw=0.5)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", output_png)

def standardize_phase1_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    canonical = {
        "globalindex": "global_index",
        "aorid": "aor_id",
        "visitlabel": "visit_label",
        "visitindex": "visit_index",
        "segmentid": "segment_id",
        "bjdutc": "bjd_utc",
        "fluxraw": "flux_raw",
        "fluxnormvisit": "flux_norm_visit",
        "fluxnormglobal": "flux_norm_global",
        "xcent": "x_cent",
        "ycent": "y_cent",
        "frameok": "frame_ok",
    }
    rename = {}
    for column in df.columns:
        compact = column.lower().replace("_", "").replace(" ", "")
        if compact in canonical:
            rename[column] = canonical[compact]
    return df.rename(columns=rename)


def load_phase1_photometry(input_csv: Path) -> pd.DataFrame:
    df = standardize_phase1_columns(pd.read_csv(input_csv))
    required = ["bjd_utc", "flux_norm_global", "x_cent", "y_cent", "beta"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Phase 1 CSV is missing required columns: {missing}")

    if "global_index" not in df.columns:
        df["global_index"] = np.arange(len(df), dtype=int)
    if "aor_id" not in df.columns:
        df["aor_id"] = "all"
    if "visit_label" not in df.columns:
        df["visit_label"] = "unknown"
    if "frame_ok" in df.columns:
        df = df.loc[as_bool_mask(df["frame_ok"])].copy()

    finite = np.ones(len(df), dtype=bool)
    for column in required:
        finite &= np.isfinite(df[column].to_numpy(float))
    finite &= df["flux_norm_global"].to_numpy(float) > 0.0
    df = df.loc[finite].sort_values(["bjd_utc", "global_index"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No usable Phase 1 frames remain")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HAT-P-2b Spitzer/IRAC 4.5 um Phase 2 analysis without pulsations"
    )
    parser.add_argument(
        "--phase1-csv",
        type=Path,
        default=Path("output/phase1_hatp2b_45um_photometry.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--prefix", default="phase2_hatp2b_45um")
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--outer-iterations", type=int, default=DEFAULT_OUTER_ITERATIONS)
    parser.add_argument("--walkers", type=int, default=DEFAULT_N_WALKERS)
    parser.add_argument("--steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--burn", type=int, default=DEFAULT_N_BURN)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-emcee", action="store_true")
    parser.add_argument(
        "--no-reference-priors",
        action="store_true",
        help="Use bounded physical priors only; not recommended for the first run",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip corner and MCMC convergence plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not 0.0 < args.bin_width < 1.0:
        raise ValueError("--bin-width must be between 0 and 1")
    if args.outer_iterations < 1:
        raise ValueError("--outer-iterations must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_dir / args.prefix
    use_reference_priors = not args.no_reference_priors

    validate_eclipse_window()
    df = load_phase1_photometry(args.phase1_csv)
    df["phase"] = phase_fold_01(df["bjd_utc"].to_numpy(float))
    LOG.info("Loaded %d frames in %d AORs", len(df), df["aor_id"].nunique())
    LOG.info(
        "Geometry: P=%.7f d, e=%.5f, omega=%.2f deg, i=%.2f deg, a/Rs=%.4f",
        P_ORB,
        ECC,
        OMEGA_DEG,
        INC_DEG,
        A_RS,
    )
    LOG.info(
        "Display phases: transit=%.5f, periastron=%.5f, occultation=%.5f",
        PHASE_TRANSIT,
        PHASE_PERIASTRON,
        PHASE_OCCULTATION,
    )

    theta_reference = THETA_DEWIT.copy()
    corrected = None
    binned = None

    for iteration in range(args.outer_iterations):
        LOG.info("Outer correction iteration %d/%d", iteration + 1, args.outer_iterations)
        corrected = correct_photometry(df, theta_reference)
        binned = bin_phase_curve(
            corrected["phase"].to_numpy(float),
            corrected["flux_corr_final"].to_numpy(float),
            corrected["aor_id"].to_numpy(),
            args.bin_width,
        )
        if len(binned) < 30:
            raise RuntimeError(f"Only {len(binned)} populated phase bins")
        theta_reference = fit_map(binned, theta_reference, use_reference_priors)

    LOG.info("Final fixed-point correction before science fit")
    corrected = correct_photometry(df, theta_reference)
    binned = bin_phase_curve(
        corrected["phase"].to_numpy(float),
        corrected["flux_corr_final"].to_numpy(float),
        corrected["aor_id"].to_numpy(),
        args.bin_width,
    )
    theta_reference = fit_map(binned, theta_reference, use_reference_priors)

    if args.skip_emcee:
        result = FitResult(theta_map=theta_reference)
    else:
        result = run_emcee_fit(
            binned,
            theta_reference,
            args.walkers,
            args.steps,
            args.burn,
            args.seed,
            use_reference_priors,
        )

    figure_path = output_stem.with_name(output_stem.name + "_fig1_no_pulsations.png")
    plot_dewit_fig1_style(binned, result.theta_map, figure_path)

    if result.sampler is not None and not args.skip_diagnostics:
        statistics = mcmc_statistics(result)
        trace_path = output_stem.with_name(output_stem.name + "_mcmc_diagnostics.png")
        corner_path = output_stem.with_name(output_stem.name + "_corner.png")
        plot_mcmc_traces_and_convergence(result, statistics, trace_path)
        plot_corner_posterior(result, statistics, corner_path, seed=args.seed)

    LOG.info("Final science plot: %s", figure_path)



if __name__ == "__main__":
    main()