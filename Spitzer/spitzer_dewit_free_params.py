from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import emcee
import batman 

P_ORB = 5.6334675         
T0_TRANSIT = 2455288.84969 
ECC = 0.51023         
OMEGA_DEG = 188.44   

DEFAULT_BIN_WIDTH = 0.00025

PHASE_MIN_PPM = 322.0
PHASE_PEAK_PPM = 1178.0
PHASE_PEAK_OFFSET_HR = 5.40
PHASE_RISE_HR = 5.5
PHASE_DECAY_HR = 10.3

BASE_PHOT_NOISE_PPM = 75.0   # typical noise per 1 hr bin

RP_RS = 0.0704        
A_RS = 8.28     
INC_DEG = 85.0            
LIMB_DARKENING_COEFFS = [0.12, 0.34, 0.20, 0.10]  


def mad_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad):
        return np.nan
    return 1.4826 * mad

def phase_fold(bjd: np.ndarray,
               period: float = P_ORB,
               t0: float = T0_TRANSIT) -> np.ndarray:
    ph = ((np.asarray(bjd, dtype=float) - t0) / period) % 1.0
    ph = np.where(ph > 0.5, ph - 1.0, ph)
    return ph


def bin_phase(phase: np.ndarray,
              flux: np.ndarray,
              bin_width: float = DEFAULT_BIN_WIDTH) -> pd.DataFrame:
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)

    edges = np.arange(-0.5, 0.5 + bin_width, bin_width)
    idx = np.digitize(phase, edges) - 1

    rows = []
    for k in range(len(edges) - 1):
        m = idx == k
        if not np.any(m):
            continue
        ff = flux[m]
        pp = phase[m]
        good = np.isfinite(ff) & np.isfinite(pp)
        if good.sum() == 0:
            continue
        ff = ff[good]
        pp = pp[good]
        n = ff.size
        med = np.nanmedian(ff)
        ebin = mad_std(ff) / np.sqrt(max(n, 1))
        rows.append(
            {
                "phase_center": 0.5 * (edges[k] + edges[k + 1]),
                "phase_median": float(np.nanmedian(pp)),
                "flux_median": float(med),
                "flux_err": float(ebin),
                "n_points": int(n),
            }
        )
    return pd.DataFrame(rows)

def time_of_periastron(t0_bjd: float,
                       period_days: float,
                       e: float,
                       omega_deg: float) -> float:
    omega = np.deg2rad(omega_deg)
    f_tr = (0.5 * np.pi) - omega

    tan_half_E = np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(0.5 * f_tr)
    E_tr = 2.0 * np.arctan(tan_half_E)
    M_tr = E_tr - e * np.sin(E_tr)

    delta_t_days = (period_days / (2.0 * np.pi)) * M_tr
    t_peri = t0_bjd - delta_t_days
    return float(t_peri)


def asymmetric_lorentzian_flux_param(t_bjd: np.ndarray,
                                     t_peri_bjd: float,
                                     F_min_ppm: float,
                                     F_peak_ppm: float,
                                     t_peak_hr: float,
                                     tau_rise_hr: float,
                                     tau_decay_hr: float) -> np.ndarray:
    t_bjd = np.asarray(t_bjd, dtype=float)
    delta_t_hr = (t_bjd - t_peri_bjd) * 24.0

    F_min = F_min_ppm
    A = F_peak_ppm - F_min_ppm
    t_peak = t_peak_hr
    tau_rise = tau_rise_hr
    tau_decay = tau_decay_hr

    u = np.empty_like(delta_t_hr)
    before = delta_t_hr <= t_peak
    after = ~before
    u[before] = (delta_t_hr[before] - t_peak) / tau_rise
    u[after] = (delta_t_hr[after] - t_peak) / tau_decay

    return F_min + A / (1.0 + u**2)


def fixed_phase_model_flux(t_bjd: np.ndarray) -> np.ndarray:
    t_bjd = np.asarray(t_bjd, dtype=float)
    t_peri = time_of_periastron(T0_TRANSIT, P_ORB, ECC, OMEGA_DEG)
    phase_ppm = asymmetric_lorentzian_flux_param(
        t_bjd=t_bjd,
        t_peri_bjd=t_peri,
        F_min_ppm=PHASE_MIN_PPM,
        F_peak_ppm=PHASE_PEAK_PPM,
        t_peak_hr=PHASE_PEAK_OFFSET_HR,
        tau_rise_hr=PHASE_RISE_HR,
        tau_decay_hr=PHASE_DECAY_HR,
    )
    return 1.0 + phase_ppm / 1e6

def make_transit_params() -> batman.TransitParams:
    params = batman.TransitParams()
    params.t0 = T0_TRANSIT
    params.per = P_ORB
    params.rp = RP_RS
    params.a = A_RS
    params.inc = INC_DEG
    params.ecc = ECC
    params.w = OMEGA_DEG
    params.limb_dark = "nonlinear"
    params.u = LIMB_DARKENING_COEFFS
    return params


def batman_transit_flux(time_bjd: np.ndarray) -> np.ndarray:
    params = make_transit_params()
    exposure_days = 0.4 / 86400.0
    model = batman.TransitModel(
        params,
        time_bjd,
        supersample_factor=10,
        exp_time=exposure_days,
    )
    return model.light_curve(params)


def eclipse_visibility(time_bjd: np.ndarray) -> np.ndarray:
    params = make_transit_params()

    omega = np.deg2rad(OMEGA_DEG)
    f_occ = 1.5 * np.pi - omega
    tan_half_E_occ = np.sqrt((1.0 - ECC) / (1.0 + ECC)) * np.tan(0.5 * f_occ)
    E_occ = 2.0 * np.arctan(tan_half_E_occ)
    M_occ = E_occ - ECC * np.sin(E_occ)
    delta_t_occ_days = (P_ORB / (2.0 * np.pi)) * M_occ
    t_secondary = T0_TRANSIT + delta_t_occ_days
    params.t_secondary = t_secondary

    params.fp = 1.0
    params.limb_dark = "uniform"
    params.u = []

    exposure_days = 0.4 / 86400.0
    model = batman.TransitModel(
        params,
        time_bjd,
        transittype="secondary",
        supersample_factor=10,
        exp_time=exposure_days,
    )
    visibility = model.light_curve(params) - 1.0
    return np.clip(visibility, 0.0, 1.0)

def system_model_flux(t_bjd: np.ndarray,
                      theta_phase: np.ndarray) -> np.ndarray:
    t_bjd = np.asarray(t_bjd, dtype=float)

    # Stellar transit
    transit_flux = batman_transit_flux(t_bjd)

    # Planet heating
    F_min_ppm, F_peak_ppm, t_peak_hr, tau_rise_hr, tau_decay_hr = theta_phase
    t_peri = time_of_periastron(T0_TRANSIT, P_ORB, ECC, OMEGA_DEG)
    planet_ppm = asymmetric_lorentzian_flux_param(
        t_bjd=t_bjd,
        t_peri_bjd=t_peri,
        F_min_ppm=F_min_ppm,
        F_peak_ppm=F_peak_ppm,
        t_peak_hr=t_peak_hr,
        tau_rise_hr=tau_rise_hr,
        tau_decay_hr=tau_decay_hr,
    )

    # Eclipse visibility
    vis = eclipse_visibility(t_bjd)

    return transit_flux * (1.0 + planet_ppm * vis / 1e6)

N_IP_NEIGHBORS = 50
MIN_KERNEL_WIDTH_PIX = 0.002
IP_QUERY_CHUNK_SIZE = 25_000

def weighted_mean(values: np.ndarray,
                  weights: np.ndarray) -> float:
    total_weight = np.sum(weights)
    if (not np.isfinite(total_weight)) or total_weight <= 0.0:
        return np.nan
    return np.sum(values * weights) / total_weight


def build_ip_sensitivity_map(x_cent: np.ndarray,
                             y_cent: np.ndarray,
                             detector_ratio: np.ndarray) -> np.ndarray:
    x_cent = np.asarray(x_cent, dtype=float)
    y_cent = np.asarray(y_cent, dtype=float)
    detector_ratio = np.asarray(detector_ratio, dtype=float)

    n_points = len(x_cent)
    if n_points <= N_IP_NEIGHBORS:
        raise RuntimeError("Not enough rows for the requested IP map.")

    coords = np.column_stack([x_cent, y_cent])
    tree = cKDTree(coords)
    query_k = N_IP_NEIGHBORS + 1

    sensitivity = np.ones(n_points, dtype=float)

    print("\nBuilding intrapixel sensitivity map (cKDTree leave-one-out)...")

    for start in range(0, n_points, IP_QUERY_CHUNK_SIZE):
        stop = min(start + IP_QUERY_CHUNK_SIZE, n_points)
        _, neighbour_indices = tree.query(
            coords[start:stop],
            k=query_k,
            workers=-1,
        )
        if neighbour_indices.ndim == 1:
            neighbour_indices = neighbour_indices[:, None]

        for local_index, global_index in enumerate(range(start, stop)):
            neighbours = neighbour_indices[local_index]
            neighbours = neighbours[neighbours != global_index]
            neighbours = neighbours[:N_IP_NEIGHBORS]

            if len(neighbours) < 5:
                sensitivity[global_index] = 1.0
                continue

            dx = x_cent[neighbours] - x_cent[global_index]
            dy = y_cent[neighbours] - y_cent[global_index]

            sigma_x = max(np.std(dx, ddof=1), MIN_KERNEL_WIDTH_PIX)
            sigma_y = max(np.std(dy, ddof=1), MIN_KERNEL_WIDTH_PIX)

            weights = np.exp(
                -0.5 * ((dx / sigma_x)**2 + (dy / sigma_y)**2)
            )

            estimate = weighted_mean(detector_ratio[neighbours], weights)
            if np.isfinite(estimate) and estimate > 0.0:
                sensitivity[global_index] = estimate
            else:
                sensitivity[global_index] = 1.0

        print(f" IP map progress: {stop:,}/{n_points:,} "
              f"({100.0 * stop / n_points:5.1f}%)")

    valid = np.isfinite(sensitivity) & (sensitivity > 0.0)
    sensitivity /= np.median(sensitivity[valid])

    return sensitivity


def apply_ip_correction(flux_global: np.ndarray,
                        fixed_model: np.ndarray,
                        x_cent: np.ndarray,
                        y_cent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flux_global = np.asarray(flux_global, dtype=float)
    fixed_model = np.asarray(fixed_model, dtype=float)
    x_cent = np.asarray(x_cent, dtype=float)
    y_cent = np.asarray(y_cent, dtype=float)

    detector_ratio = flux_global / fixed_model
    detector_ratio /= np.median(detector_ratio)

    sensitivity = build_ip_sensitivity_map(
        x_cent=x_cent,
        y_cent=y_cent,
        detector_ratio=detector_ratio,
    )

    corrected_flux = flux_global / sensitivity
    global_norm = np.median(corrected_flux / fixed_model)
    corrected_flux /= global_norm

    return corrected_flux, sensitivity

def log_likelihood_theta(theta: np.ndarray,
                         phase: np.ndarray,
                         flux: np.ndarray,
                         flux_err: np.ndarray) -> float:
    F_min_ppm, F_peak_ppm, t_peak_hr, tau_rise_hr, tau_decay_hr, log_sigma_ppm = theta

    t_bjd = T0_TRANSIT + phase * P_ORB
    model_flux = system_model_flux(
        t_bjd,
        np.array([F_min_ppm, F_peak_ppm, t_peak_hr, tau_rise_hr, tau_decay_hr]),
    )

    sigma_ppm = np.exp(log_sigma_ppm)
    total_err = np.sqrt(flux_err**2 + (sigma_ppm / 1e6)**2)

    resid = flux - model_flux
    chi2 = np.sum((resid / total_err)**2)
    norm = np.sum(np.log(2.0 * np.pi * total_err**2))
    return -0.5 * (chi2 + norm)


def log_prior_theta(theta: np.ndarray) -> float:
    F_min_ppm, F_peak_ppm, t_peak_hr, tau_rise_hr, tau_decay_hr, log_sigma_ppm = theta

    if not (0.0 < F_min_ppm < 2000.0):
        return -np.inf
    if not (0.0 < F_peak_ppm < 3000.0):
        return -np.inf
    if not (0.0 < t_peak_hr < 20.0):
        return -np.inf
    if not (0.5 < tau_rise_hr < 20.0):
        return -np.inf
    if not (0.5 < tau_decay_hr < 40.0):
        return -np.inf
    if not (np.log(10.0) < log_sigma_ppm < np.log(1000.0)):
        return -np.inf
    return 0.0


def log_posterior_theta(theta: np.ndarray,
                        phase: np.ndarray,
                        flux: np.ndarray,
                        flux_err: np.ndarray) -> float:
    lp = log_prior_theta(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_theta(theta, phase, flux, flux_err)
    return lp + ll


def run_emcee_lorentzian(phase: np.ndarray,
                         flux: np.ndarray,
                         flux_err: np.ndarray,
                         n_walkers: int = 48,
                         n_steps: int = 3000,
                         burnin: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    init_F_min = PHASE_MIN_PPM
    init_F_peak = PHASE_PEAK_PPM
    init_t_peak = PHASE_PEAK_OFFSET_HR
    init_tau_rise = PHASE_RISE_HR
    init_tau_decay = PHASE_DECAY_HR
    init_log_sigma = np.log(BASE_PHOT_NOISE_PPM)

    p0 = np.vstack([
        np.random.normal(init_F_min,   50.0,  size=n_walkers),
        np.random.normal(init_F_peak,  50.0,  size=n_walkers),
        np.random.normal(init_t_peak,   1.0,  size=n_walkers),
        np.random.normal(init_tau_rise, 1.0,  size=n_walkers),
        np.random.normal(init_tau_decay,1.0,  size=n_walkers),
        np.random.normal(init_log_sigma,0.1,  size=n_walkers),
    ]).T

    sampler = emcee.EnsembleSampler(
        n_walkers,
        6,
        log_posterior_theta,
        args=(phase, flux, flux_err),
    )

    print("\nRunning EMCEE Lorentzian-parameter sampling on binned data...")
    sampler.run_mcmc(p0, n_steps, progress=True)

    chain = sampler.get_chain(discard=burnin, flat=True)
    logprob = sampler.get_log_prob(discard=burnin, flat=True)

    idx = np.argmax(logprob)
    theta_map = chain[idx]

    print("EMCEE MAP:")
    print(f"  F_min_ppm   = {theta_map[0]:.1f}")
    print(f"  F_peak_ppm  = {theta_map[1]:.1f}")
    print(f"  t_peak_hr   = {theta_map[2]:.2f}")
    print(f"  tau_rise_hr = {theta_map[3]:.2f}")
    print(f"  tau_decay_hr= {theta_map[4]:.2f}")
    print(f"  sigma_ppm   = exp({theta_map[5]:.2f}) = {np.exp(theta_map[5]):.1f} ppm")

    return chain, theta_map

def plot_fig1_style(df: pd.DataFrame,
                    binned: pd.DataFrame,
                    theta_map: np.ndarray,
                    outprefix: str) -> None:
    outprefix = Path(outprefix)

    phase_b = binned["phase_center"].to_numpy(dtype=float)
    flux_b = binned["flux_median"].to_numpy(dtype=float)

    # Model on fine phase grid using MAP parameters.
    phase_grid = np.linspace(-0.5, 0.5, 4000)
    t_grid = T0_TRANSIT + phase_grid * P_ORB
    model_flux = system_model_flux(
        t_grid,
        theta_map[:5],
    )

    # Transit center.
    phase_trans_center = 0.0

    # Occultation center from geometry.
    omega = np.deg2rad(OMEGA_DEG)
    f_occ = 1.5 * np.pi - omega
    tan_half_E_occ = np.sqrt((1.0 - ECC) / (1.0 + ECC)) * np.tan(0.5 * f_occ)
    E_occ = 2.0 * np.arctan(tan_half_E_occ)
    M_occ = E_occ - ECC * np.sin(E_occ)
    phase_occ_center = ((M_occ / (2.0 * np.pi)) % 1.0)
    if phase_occ_center > 0.5:
        phase_occ_center -= 1.0

    # Matplotlib style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.0), dpi=160, sharex=False)

    # Panel A: full phase curve.
    ax = axes[0]
    ax.plot(phase_b, flux_b, "k.", ms=3.0, alpha=0.9)
    ax.plot(phase_grid, model_flux, color="green", lw=1.2, alpha=0.9)
    ax.axhline(1.0, ls="--", lw=0.9, c="0.6")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("Relative flux")
    ax.text(0.02, 0.92, "A", transform=ax.transAxes, fontweight="bold")
    ax.grid(alpha=0.15)

    # Panel B: transit zoom.
    ax = axes[1]
    ax.plot(phase_b, flux_b, "k.", ms=3.0, alpha=0.9)
    ax.plot(phase_grid, model_flux, color="green", lw=1.2, alpha=0.9)
    ax.axhline(1.0, ls="--", lw=0.9, c="0.6")
    ax.set_xlim(phase_trans_center - 0.08, phase_trans_center + 0.08)
    ax.set_ylabel("Relative flux")
    ax.text(0.02, 0.92, "B", transform=ax.transAxes, fontweight="bold")
    ax.grid(alpha=0.15)

    # Panel C: occultation zoom.
    ax = axes[2]
    ax.plot(phase_b, flux_b, "k.", ms=3.0, alpha=0.9)
    ax.plot(phase_grid, model_flux, color="green", lw=1.2, alpha=0.9)
    ax.axhline(1.0, ls="--", lw=0.9, c="0.6")
    ax.set_xlim(phase_occ_center - 0.08, phase_occ_center + 0.08)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Relative flux")
    ax.text(0.02, 0.92, "C", transform=ax.transAxes, fontweight="bold")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    # outpng = outprefix.with_suffix("_phase3_dewit_fig1.png")
    # fig.savefig(outpng, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)

    print(f"Saved Phase 3 Figure 1-style plot")

def main():
    ap = argparse.ArgumentParser(
        description="Phase 3: de Wit 4.5 μm HAT-P-2b Figure 1-style reproduction."
    )
    ap.add_argument(
        "--input",
        default="output/phase1_hatp2b_45um_photometry.csv",
        help="Phase 1 photometry CSV.",
    )
    ap.add_argument(
        "--output-prefix",
        default="output/phase3_hatp2b_45um",
        help="Prefix for Phase 3 outputs (Figure 1-style plot).",
    )
    ap.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_BIN_WIDTH,
        help="Phase bin width (default 0.00025).",
    )
    ap.add_argument(
        "--skip-emcee",
        action="store_true",
        help="Skip EMCEE and use fixed de Wit Lorentzian model.",
    )
    args = ap.parse_args()

    outprefix = Path(args.output_prefix)
    outprefix.parent.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 CSV.
    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]

    # Normalize column names.
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "aorid":
            rename_map[c] = "aor_id"
        elif cl == "globalindex":
            rename_map[c] = "global_index"
        elif cl == "bjdutc":
            rename_map[c] = "bjd_utc"
        elif cl in {"xcent", "x_cent"}:
            rename_map[c] = "x_cent"
        elif cl in {"ycent", "y_cent"}:
            rename_map[c] = "y_cent"
        elif cl == "fluxraw":
            rename_map[c] = "flux_raw"
        elif cl == "fluxnormglobal":
            rename_map[c] = "flux_norm_global"
        elif cl == "fluxnormvisit":
            rename_map[c] = "flux_norm_visit"
        elif cl == "frameok":
            rename_map[c] = "frame_ok"
        elif cl == "apertureradius":
            rename_map[c] = "aperture_radius"
        elif cl == "nhotpixfixedframe":
            rename_map[c] = "n_hotpix_fixed_frame"
        elif cl == "beta":
            rename_map[c] = "beta"
        elif cl == "segmentid":
            rename_map[c] = "segment_id"

    df = df.rename(columns=rename_map)

    required_cols = ["bjd_utc", "flux_norm_global", "x_cent", "y_cent"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required Phase 1 column: {col}")

    if "global_index" not in df.columns:
        df["global_index"] = np.arange(len(df), dtype=int)

    df = df.sort_values(["bjd_utc", "global_index"]).reset_index(drop=True)

    # Respect Phase 1 frame_ok (no new trimming).
    if "frame_ok" in df.columns:
        frame_ok = df["frame_ok"].astype(str).str.lower().isin(
            ["true", "1", "t", "yes"]
        ) | (df["frame_ok"] == 1)
        df = df.loc[frame_ok].copy().reset_index(drop=True)

    # Drop non-finite flux_norm_global.
    df = df[np.isfinite(df["flux_norm_global"])].copy().reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No data left after Phase 1 frame_ok selection.")

    # Fixed Lorentzian-only phase model for IP map (planet only, no transit).
    t_bjd = df["bjd_utc"].to_numpy(dtype=float)
    fixed_model = fixed_phase_model_flux(t_bjd)

    # Apply Lewis-style intrapixel correction on flux_norm_global.
    flux_global = df["flux_norm_global"].to_numpy(dtype=float)
    corrected_flux, sensitivity = apply_ip_correction(
        flux_global=flux_global,
        fixed_model=fixed_model,
        x_cent=df["x_cent"].to_numpy(dtype=float),
        y_cent=df["y_cent"].to_numpy(dtype=float),
    )
    df["ip_sensitivity"] = sensitivity
    df["flux_corr_final"] = corrected_flux

    # Phase-fold
    df["phase"] = phase_fold(df["bjd_utc"].to_numpy(), period=P_ORB, t0=T0_TRANSIT)

    # Bin (binned is what we plot and fit)
    binned = bin_phase(
        df["phase"].to_numpy(),
        df["flux_corr_final"].to_numpy(),
        bin_width=args.bin_width,
    )

    if len(binned) < 10:
        raise RuntimeError("Too few binned points for EMCEE / plotting.")

    # EMCEE on Lorentzian+transit model (binned data).
    if args.skip_emcee:
        theta_map = np.array([
            PHASE_MIN_PPM,
            PHASE_PEAK_PPM,
            PHASE_PEAK_OFFSET_HR,
            PHASE_RISE_HR,
            PHASE_DECAY_HR,
            np.log(BASE_PHOT_NOISE_PPM),
        ])
    else:
        phase_b = binned["phase_center"].to_numpy(dtype=float)
        flux_b = binned["flux_median"].to_numpy(dtype=float)
        err_b = binned["flux_err"].to_numpy(dtype=float)
        err_b[~np.isfinite(err_b)] = BASE_PHOT_NOISE_PPM / 1e6
        _, theta_map = run_emcee_lorentzian(
            phase=phase_b,
            flux=flux_b,
            flux_err=err_b,
            n_walkers=48,
            n_steps=3000,
            burnin=1500,
        )

    # Single Figure 1-style plot.
    plot_fig1_style(df, binned, theta_map, str(outprefix))


if __name__ == "__main__":
    main()