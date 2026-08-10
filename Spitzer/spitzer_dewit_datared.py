#!/usr/bin/env python3
"""
Phase 1 photometry extraction for the de Wit et al. HAT-P-2b Spitzer/IRAC 4.5 um dataset.

This script combines:

  - Lewis-style image processing and ramp handling:
        * background estimation via iterative sigma-clipped Gaussian fit,
        * transient hot-pixel repair across each 64-frame cube,
        * flux-weighted centroiding in a central 3.5-pixel aperture,
        * noise-pixel beta calculation,
        * Lewis-style outlier rejection using a 16-point moving median,
        * Lewis-style ramp handling: identify continuous time segments and
          trim the first hour of each segment (standard 4.5 um ramp mitigation).

  - de Wit-style 4.5 um data set and aperture philosophy:
        * all 28 AORs listed in de Wit Table 1 (phase curve, transits,
          occultations),
        * time-varying apertures with radius r = sqrt(beta) + offset,
          with per-AOR offsets optimized on a grid to minimize residual scatter
          after moving-median detrending (as in your `spitzer_phase1_fixed.py`).

  - spitzer_phase1_fixed-style design and outputs:
        * safer subarray timing reconstruction based on MBJD_OBS and
          FRAMTIME/AINTBEG/ATIMEEND,
        * per-AOR aperture-offset optimization,
        * per-visit normalization (flux_norm_visit),
        * campaign-wide normalization (flux_norm_global),
        * per-file and per-AOR summary tables,
        * manifest JSON with timing diagnostics,
        * four-panel diagnostic plot:
              (a) x centroid,
              (b) y centroid,
              (c) noise-pixel beta,
              (d) globally normalized relative flux, after ramp trimming
                  and centroid/beta outlier clipping.

Phase 1 therefore outputs an “instrumentally cleaned but still astrophysical”
time series that already has:

  - hot pixels repaired,
  - ramp onset removed via first-hour trimming per segment,
  - AOR-specific noise-pixel apertures,
  - robust outlier clipping in flux and in centroid/beta,

but does NOT:

  - build an intrapixel sensitivity map,
  - apply intrapixel corrections,
  - fit explicit exponential ramp functions (those belong in Phase 2/3 if needed),
  - fit transit, eclipse, phase curve, or pulsation models,
  - run EMCEE.

Downstream phases (2+) can use the CSVs from this script as their starting
photometry products.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Default de Wit 4.5 um AOR table (Table 1)
DEFAULT_DEWIT_AOR_TABLE = [
    {"aor": "42789632", "label": "occultation"},
    {"aor": "42789888", "label": "phase"},
    {"aor": "43962624", "label": "phase"},
    {"aor": "43962880", "label": "phase"},
    {"aor": "43963136", "label": "transit"},
    {"aor": "43963392", "label": "occultation"},
    {"aor": "46473216", "label": "occultation"},
    {"aor": "46473472", "label": "occultation"},
    {"aor": "46473728", "label": "occultation"},
    {"aor": "46474496", "label": "occultation"},
    {"aor": "46474752", "label": "occultation"},
    {"aor": "46475008", "label": "occultation"},
    {"aor": "46475264", "label": "occultation"},
    {"aor": "46475520", "label": "occultation"},
    {"aor": "46475776", "label": "occultation"},
    {"aor": "46476032", "label": "occultation"},
    {"aor": "46476288", "label": "occultation"},
    {"aor": "46476544", "label": "occultation"},
    {"aor": "46477312", "label": "phase"},
    {"aor": "46477568", "label": "phase"},
    {"aor": "46477824", "label": "occultation"},
    {"aor": "46478336", "label": "phase"},
    {"aor": "46478592", "label": "phase"},
    {"aor": "46478848", "label": "occultation"},
    {"aor": "57786880", "label": "transit"},
    {"aor": "57787136", "label": "transit"},
    {"aor": "57787392", "label": "occultation"},
    {"aor": "57787648", "label": "occultation"},
]

# IRAC subarray pattern; expects {AOR}/r{AOR}/ch2/bcd/SPITZER_I2_*_bcd.fits
DEFAULT_GLOB = "*/r*/ch2/bcd/SPITZER_I2_*_bcd.fits"

FRAME_SHAPE = (32, 32)
CUBE_LEN = 64

SIGMA_HOTPIX = 4.5
SIGMA_OUTLIER = 4.5
MOVING_MEDIAN_WIDTH = 16

BACKGROUND_RMIN = 10.0
CENTROID_R = 3.5

# Grid of aperture offsets (additive term to sqrt(beta))
APERTURE_R_GRID = np.round(np.arange(-2.50, -1.00 + 1e-9, 0.05), 2)
APERTURE_R_FLOOR = 1.30
APERTURE_R_CEIL = 3.00

# Lewis/de Wit-style segment handling
SEGMENT_GAP_HOURS = 1.0       # identify new segments from ~1 hr gaps
TRIM_FIRST_HOUR = True        # trim first hour of each continuous segment
TRIM_HOURS = 1.0

EXPECTED_AOR_COUNT = 28
EXPECTED_TOTAL_HOURS_RANGE = (330.0, 370.0)

DIAG_FIGURE_NAME = "phase1_hatp2b_45um_diag.png"


# -------------------------------------------------------------------------
# Low-level utilities
# -------------------------------------------------------------------------

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else np.nanstd(x)


def moving_median(a: np.ndarray, width: int) -> np.ndarray:
    s = pd.Series(a)
    return s.rolling(
        width,
        center=True,
        min_periods=max(5, width // 4),
    ).median().to_numpy()


def extract_aor_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("r") and part[1:].isdigit():
            return part[1:]
    raise ValueError(f"Could not infer AOR from path: {path}")


def get_header_value(header, *keys, default=None):
    for key in keys:
        if key in header:
            return header[key]
    return default


def circular_mask(shape, x0, y0, r):
    yy, xx = np.indices(shape, dtype=float)
    return (xx - x0) ** 2 + (yy - y0) ** 2 <= r ** 2


def fractional_circular_mask(shape, x0, y0, r, sub=8):
    yy, xx = np.indices(shape, dtype=float)
    frac = np.zeros(shape, dtype=float)
    offs = (np.arange(sub, dtype=float) + 0.5) / sub - 0.5

    for dy in offs:
        for dx in offs:
            xs = xx + dx
            ys = yy + dy
            frac += (
                ((xs - x0) ** 2 + (ys - y0) ** 2) <= r ** 2
            ).astype(float)

    frac /= float(sub * sub)
    return frac


# -------------------------------------------------------------------------
# Timing reconstruction (Lewis-style, subarray cubes)
# -------------------------------------------------------------------------

def compute_bjd_utc_for_cube(header, n_frames: int):
    """
    Lewis-style frame timing reconstruction for IRAC subarray cubes.

    MBJD_OBS/BMJD_OBS is the start time of the first image in MJD.
    Frames are assumed uniformly spaced over the interval defined by
    AINTBEG and ATIMEEND. Returned timestamps are mid-exposure JD
    in a BJD_UTC-like convention.
    """
    mjd0 = get_header_value(header, "MBJD_OBS", "BMJD_OBS")
    if mjd0 is None:
        raise KeyError("Missing MBJD_OBS/BMJD_OBS in FITS header")

    aintbeg = get_header_value(header, "AINTBEG")
    atimeend = get_header_value(header, "ATIMEEND")
    framtime = float(get_header_value(header, "FRAMTIME", "EXPTIME", default=0.4))

    timing_mode = "mbjd_plus_header_span"
    warning = ""

    if aintbeg is not None and atimeend is not None:
        total = float(atimeend) - float(aintbeg)
        expected = n_frames * framtime

        if (not np.isfinite(total)) or total <= 0:
            total = expected
            timing_mode = "mbjd_plus_framtime_fallback"
            warning = "Non-positive ATIMEEND-AINTBEG; fell back to FRAMTIME"
        else:
            if abs(total - expected) > max(1.0, 0.25 * expected):
                warning = (
                    f"Header span differs from n_frames*FRAMTIME: "
                    f"{total:.6f}s vs {expected:.6f}s"
                )
        dt = total / n_frames
    else:
        dt = framtime
        total = n_frames * dt
        expected = total
        timing_mode = "mbjd_plus_framtime_fallback"
        warning = "Missing AINTBEG/ATIMEEND; fell back to FRAMTIME"

    mids = (np.arange(n_frames, dtype=float) + 0.5) * dt
    jd_mid = float(mjd0) + mids / 86400.0 + 2400000.5

    timing_meta = {
        "timing_mode": timing_mode,
        "timing_anchor_mjd": float(mjd0),
        "framtime_s": float(framtime),
        "aintbeg_s": float(aintbeg) if aintbeg is not None else np.nan,
        "atimeend_s": float(atimeend) if atimeend is not None else np.nan,
        "header_span_s": float(total),
        "expected_span_s": float(expected),
        "span_minus_expected_s": float(total - expected),
        "timing_warning": warning,
    }
    return jd_mid, timing_meta


# -------------------------------------------------------------------------
# Image-level processing
# -------------------------------------------------------------------------

def estimate_background(image: np.ndarray, rmin: float = BACKGROUND_RMIN) -> float:
    """
    Lewis-style background estimation: pixels beyond rmin from the PSF center,
    iterative 3-sigma clipping and Gaussian fit to the histogram.
    """
    ny, nx = image.shape
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0

    yy, xx = np.indices(image.shape, dtype=float)
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    vals = image[rr > rmin].astype(float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 50:
        return float(np.nanmedian(vals)) if vals.size else 0.0

    clipped = vals.copy()
    for _ in range(3):
        mu = np.nanmedian(clipped)
        sig = robust_std(clipped)
        if not np.isfinite(sig) or sig <= 0:
            break
        keep = np.abs(clipped - mu) < 3.0 * sig
        if keep.sum() < 50:
            break
        clipped = clipped[keep]

    hist, edges = np.histogram(clipped, bins="fd")
    centers = 0.5 * (edges[:-1] + edges[1:])
    if hist.sum() == 0:
        return float(np.nanmedian(clipped))

    p0 = [
        hist.max(),
        np.nanmedian(clipped),
        robust_std(clipped) or np.nanstd(clipped) or 1.0,
    ]
    try:
        popt, _ = curve_fit(gaussian, centers, hist, p0=p0, maxfev=2000)
        return float(popt[1])
    except Exception:
        return float(np.average(centers, weights=hist))


def repair_hot_pixels_cube(cube: np.ndarray, sigma: float = SIGMA_HOTPIX):
    """
    Transient hot-pixel repair across a cube: flag values > sigma * std
    at each pixel position over the 64 frames, replace with that pixel's median.
    """
    cube = np.asarray(cube, dtype=float)
    med = np.nanmedian(cube, axis=0)
    std = np.nanstd(cube, axis=0)
    std[~np.isfinite(std) | (std == 0)] = np.inf
    resid = cube - med
    bad = np.abs(resid) > sigma * std
    repaired = cube.copy()
    repaired[bad] = np.broadcast_to(med, cube.shape)[bad]
    return repaired, bad


def flux_weighted_centroid(image: np.ndarray, r: float = CENTROID_R):
    """
    Flux-weighted centroid within a central aperture of radius r.
    """
    ny, nx = image.shape
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0
    mask = circular_mask(image.shape, x0, y0, r)

    sub = np.where(mask, np.clip(image, 0.0, None), 0.0)
    total = np.nansum(sub)
    if not np.isfinite(total) or total <= 0:
        return x0, y0

    yy, xx = np.indices(image.shape, dtype=float)
    x = np.nansum(xx * sub) / total
    y = np.nansum(yy * sub) / total
    return float(x), float(y)


def noise_pixel_beta(image: np.ndarray, x: float, y: float, r: float = CENTROID_R) -> float:
    """
    IRAC noise-pixel parameter:

        beta = (sum I)^2 / sum I^2

    within an aperture of radius r centered at (x, y).
    """
    mask = circular_mask(image.shape, x, y, r)
    vals = image[mask].astype(float)
    vals = vals[np.isfinite(vals)]
    s1 = np.nansum(vals)
    s2 = np.nansum(vals ** 2)
    if s2 <= 0:
        return np.nan
    return float((s1 ** 2) / s2)


def aperture_flux(image: np.ndarray, x: float, y: float, r: float) -> float:
    frac = fractional_circular_mask(image.shape, x, y, r, sub=8)
    return float(np.nansum(image * frac))


def aperture_radius_from_beta(beta: float, offset: float) -> float:
    """
    De Wit-style time-varying aperture:

        r = sqrt(beta) + offset,

    clipped to [APERTURE_R_FLOOR, APERTURE_R_CEIL].
    """
    if not np.isfinite(beta) or beta <= 0:
        return np.nan
    r = math.sqrt(beta) + offset
    return float(np.clip(r, APERTURE_R_FLOOR, APERTURE_R_CEIL))


# -------------------------------------------------------------------------
# Data classes for summaries
# -------------------------------------------------------------------------

@dataclass
class FileSummary:
    aor_id: str
    fits_file: str
    n_frames: int
    n_hotpix_fixed: int
    bjd_start: float
    bjd_end: float
    timing_mode: str
    framtime_s: float
    aintbeg_s: float
    atimeend_s: float
    header_span_s: float
    expected_span_s: float
    span_minus_expected_s: float
    timing_warning: str


@dataclass
class AORSummary:
    aor_id: str
    label: str
    n_files: int
    n_frames_raw: int
    n_frames_kept: int
    bjd_start: float
    bjd_end: float
    duration_hr: float
    beta_median: float
    radius_median: float
    x_median: float
    y_median: float
    aperture_offset: float
    flux_scatter_ppm: float


# -------------------------------------------------------------------------
# Per-file processing
# -------------------------------------------------------------------------

def process_bcd_file(path: Path):
    """
    Process one subarray BCD cube into a per-frame DataFrame + FileSummary.

    The DataFrame includes BJD_UTC, background, centroid, beta, and a
    placeholder frame_ok_outlier flag; images themselves are retained in a
    temporary '_image' column for later aperture optimization.
    """
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header

    if data.ndim != 3 or data.shape[0] != CUBE_LEN or data.shape[1:] != FRAME_SHAPE:
        raise ValueError(f"Unexpected cube shape {data.shape} in {path}")

    bjd, timing_meta = compute_bjd_utc_for_cube(header, data.shape[0])
    bg = np.array([estimate_background(frame) for frame in data], dtype=float)
    bgsub = data - bg[:, None, None]
    repaired, bad = repair_hot_pixels_cube(bgsub)

    rows = []
    for i, frame in enumerate(repaired):
        x, y = flux_weighted_centroid(frame, CENTROID_R)
        beta = noise_pixel_beta(frame, x, y, CENTROID_R)
        rows.append(
            {
                "fits_file": str(path),
                "cube_index": int(i),
                "bjd_utc": float(bjd[i]),
                "background": float(bg[i]),
                "x_cent": x,
                "y_cent": y,
                "beta": beta,
                "n_hotpix_fixed_frame": int(np.count_nonzero(bad[i])),
                "frame_ok_outlier": True,
                "_image": frame,
            }
        )

    df = pd.DataFrame(rows)
    aor = extract_aor_from_path(path)
    summary = FileSummary(
        aor_id=aor,
        fits_file=str(path),
        n_frames=len(df),
        n_hotpix_fixed=int(np.count_nonzero(bad)),
        bjd_start=float(df["bjd_utc"].min()),
        bjd_end=float(df["bjd_utc"].max()),
        timing_mode=str(timing_meta["timing_mode"]),
        framtime_s=float(timing_meta["framtime_s"]),
        aintbeg_s=float(timing_meta["aintbeg_s"])
        if np.isfinite(timing_meta["aintbeg_s"])
        else np.nan,
        atimeend_s=float(timing_meta["atimeend_s"])
        if np.isfinite(timing_meta["atimeend_s"])
        else np.nan,
        header_span_s=float(timing_meta["header_span_s"])
        if np.isfinite(timing_meta["header_span_s"])
        else np.nan,
        expected_span_s=float(timing_meta["expected_span_s"]),
        span_minus_expected_s=float(timing_meta["span_minus_expected_s"])
        if np.isfinite(timing_meta["span_minus_expected_s"])
        else np.nan,
        timing_warning=str(timing_meta["timing_warning"]),
    )
    return df, summary


# -------------------------------------------------------------------------
# Lewis-style flux outlier clipping
# -------------------------------------------------------------------------

def clip_outliers(flux: np.ndarray, window_points: int = MOVING_MEDIAN_WIDTH,
                  sigma_threshold: float = SIGMA_OUTLIER) -> np.ndarray:
    """
    Lewis-style photometric clipping.

    Reject points more than sigma_threshold * robust_std from a window_points
    moving median in flux.
    """
    flux = np.asarray(flux, dtype=float)
    local_med = moving_median(flux, window_points)
    resid = flux - local_med
    sig = robust_std(resid)

    if not np.isfinite(sig) or sig <= 0:
        return np.isfinite(flux)

    return (
        np.isfinite(flux)
        & np.isfinite(resid)
        & (np.abs(resid) <= sigma_threshold * sig)
    )


# -------------------------------------------------------------------------
# Aperture offset optimization and per-AOR finalization
# -------------------------------------------------------------------------

def optimize_aperture_offset(df: pd.DataFrame, offsets: Iterable[float]) -> float:
    """
    Grid search over aperture offsets to minimize the robust std of
    flux_norm_visit residuals after moving-median detrending.

    This reproduces the behavior in your working Phase 1 script,
    providing de Wit-style AOR-dependent aperture offsets.
    """
    best_offset = None
    best_metric = np.inf

    images = df["_image"].tolist()
    x = df["x_cent"].to_numpy(float)
    y = df["y_cent"].to_numpy(float)
    beta = df["beta"].to_numpy(float)

    for off in offsets:
        flux = np.empty(len(df), dtype=float)
        flux[:] = np.nan
        for i, img in enumerate(images):
            r = aperture_radius_from_beta(beta[i], off)
            flux[i] = aperture_flux(img, x[i], y[i], r)

        med = np.nanmedian(flux)
        if not np.isfinite(med) or med <= 0:
            continue

        fn = flux / med
        loc = moving_median(fn, MOVING_MEDIAN_WIDTH)
        resid = fn - loc
        metric = robust_std(resid)

        if np.isfinite(metric) and metric < best_metric:
            best_metric = metric
            best_offset = off

    if best_offset is None:
        raise RuntimeError("Failed to optimize aperture offset")

    return float(best_offset)


def finalize_aor_photometry(df: pd.DataFrame, aperture_offset: float) -> pd.DataFrame:
    """
    Given a per-AOR DataFrame with images, centroid, and beta,
    compute aperture_radius, flux_raw, flux_norm_visit, and frame_ok_outlier.

    Outlier clipping is per-visit, based on residuals to a moving median
    in flux_norm_visit, with SIGMA_OUTLIER threshold.
    """
    images = df["_image"].tolist()

    flux = []
    radius = []
    for img, xc, yc, beta in zip(
        images,
        df["x_cent"],
        df["y_cent"],
        df["beta"],
    ):
        r = aperture_radius_from_beta(beta, aperture_offset)
        radius.append(r)
        flux.append(aperture_flux(img, xc, yc, r))

    out = df.drop(columns=["_image"]).copy()
    out["aperture_offset"] = aperture_offset
    out["aperture_radius"] = np.asarray(radius, dtype=float)
    out["flux_raw"] = np.asarray(flux, dtype=float)

    med = np.nanmedian(out["flux_raw"])
    out["flux_norm_visit"] = out["flux_raw"] / med if med > 0 else np.nan

    loc = moving_median(
        out["flux_norm_visit"].to_numpy(float),
        MOVING_MEDIAN_WIDTH,
    )
    resid = out["flux_norm_visit"].to_numpy(float) - loc
    sig = robust_std(resid)

    keep_out = (
        np.isfinite(out["flux_norm_visit"])
        & np.isfinite(out["beta"])
        & np.isfinite(out["x_cent"])
        & np.isfinite(out["y_cent"])
    )
    if np.isfinite(sig) and sig > 0:
        keep_out &= np.abs(resid) <= SIGMA_OUTLIER * sig

    out["frame_ok_outlier"] = keep_out
    return out


# -------------------------------------------------------------------------
# File discovery and AOR metadata
# -------------------------------------------------------------------------

def discover_files(base_path: Path, aor_whitelist: Optional[set[str]]) -> list[Path]:
    files = sorted(base_path.glob(DEFAULT_GLOB))
    selected = []
    for path in files:
        aor = extract_aor_from_path(path)
        if aor_whitelist is None or aor in aor_whitelist:
            selected.append(path)
    return selected


def load_aor_table(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(DEFAULT_DEWIT_AOR_TABLE)
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text()))
    return pd.read_csv(path)


# -------------------------------------------------------------------------
# Segment identification and ramp trimming (Lewis-style)
# -------------------------------------------------------------------------

def assign_time_segments(photometry: pd.DataFrame, gap_hours: float = SEGMENT_GAP_HOURS) -> pd.DataFrame:
    """
    Assign continuous observing-segment IDs from elapsed-time gaps only.

    An AOR boundary is not itself a new segment; segments are purely
    time-contiguous, matching Lewis/de Wit trimming logic across downlinks.
    """
    data = photometry.sort_values(
        ["bjd_utc", "aor_id", "fits_file", "cube_index"]
    ).copy()

    times = data["bjd_utc"].to_numpy(dtype=float)
    gap_days = gap_hours / 24.0

    segment_id = np.zeros(len(data), dtype=int)
    if len(data) > 1:
        starts_new_segment = np.diff(times) > gap_days
        segment_id[1:] = np.cumsum(starts_new_segment)

    data["segment_id"] = segment_id
    return data.reset_index(drop=True)


def first_hour_keep_mask(photometry: pd.DataFrame, trim_hours: float = TRIM_HOURS) -> np.ndarray:
    """
    Keep frames beginning trim_hours after each segment start.

    This is the Lewis/de Wit 4.5 um standard ramp correction: trim the
    first hour of each observation and each downlink/reacquisition gap.
    """
    keep = np.ones(len(photometry), dtype=bool)
    trim_days = trim_hours / 24.0

    for _, group in photometry.groupby("segment_id", sort=False):
        indices = group.index.to_numpy()
        times = group["bjd_utc"].to_numpy(dtype=float)
        segment_start = np.nanmin(times)
        keep[indices] = times >= segment_start + trim_days

    return keep


# -------------------------------------------------------------------------
# Centroid/beta clipping (new)
# -------------------------------------------------------------------------

def make_position_beta_mask(
    photometry: pd.DataFrame,
    sigma_pos: float = 5.0,
    sigma_beta: float = 5.0,
    hard_dx: float = 0.30,
    hard_dy: float = 0.30,
) -> np.ndarray:
    """
    Identify outliers in x_cent, y_cent, and beta, per (aor_id, segment_id).

    Frames are flagged as bad if they are far from the local median in
    centroid or beta, even if their flux looks fine. This removes the
    repeating vertical spikes seen in the diagnostic plots without
    touching the main trends.

    Returns a boolean mask `frame_ok_posbeta`.
    """
    mask = np.ones(len(photometry), dtype=bool)

    for (_, _), group in photometry.groupby(["aor_id", "segment_id"], sort=False):
        idx = group.index.to_numpy()
        x = group["x_cent"].to_numpy(float)
        y = group["y_cent"].to_numpy(float)
        b = group["beta"].to_numpy(float)

        # Medians
        x0 = np.nanmedian(x)
        y0 = np.nanmedian(y)
        b0 = np.nanmedian(b)

        # Robust scatters
        sx = robust_std(x)
        sy = robust_std(y)
        sb = robust_std(b)

        ok = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(b)
        )

        if np.isfinite(sx) and sx > 0:
            ok &= np.abs(x - x0) <= sigma_pos * sx
        if hard_dx is not None and np.isfinite(x0):
            ok &= np.abs(x - x0) <= hard_dx

        if np.isfinite(sy) and sy > 0:
            ok &= np.abs(y - y0) <= sigma_pos * sy
        if hard_dy is not None and np.isfinite(y0):
            ok &= np.abs(y - y0) <= hard_dy

        if np.isfinite(sb) and sb > 0:
            ok &= np.abs(b - b0) <= sigma_beta * sb

        mask[idx] = ok

    return mask


# -------------------------------------------------------------------------
# Phase 1 outputs + diagnostic plotting
# -------------------------------------------------------------------------

def make_phase1_outputs(base_path: Path, output_dir: Path, aor_table_path: Optional[Path], allow_all: bool = False):
    """
    Main Phase 1 pipeline:

      - loads or constructs the AOR table,
      - discovers matching BCD files for the selected AORs,
      - processes cubes into per-file DataFrames (Lewis-style),
      - optimizes aperture offsets per AOR (de Wit-style),
      - finalizes per-AOR photometry (flux_raw, flux_norm_visit, frame_ok_outlier),
      - concatenates all AORs,
      - identifies continuous segments and trims first hour in each (ramp correction),
      - clips centroid/beta outliers per AOR+segment,
      - builds a campaign-wide flux_norm_global from trimmed, non-outlier frames,
      - writes photometry, AOR summary, file summary, manifest,
      - makes a four-panel diagnostic plot.
    """
    aor_table = load_aor_table(aor_table_path)
    if len(aor_table) == 0 and not allow_all:
        raise RuntimeError(
            "AOR table is empty. Populate DEFAULT_DEWIT_AOR_TABLE or pass --aor-table."
        )

    if len(aor_table) > 0:
        aor_table = aor_table.copy()
        aor_table["aor"] = aor_table["aor"].astype(str)
        if "label" not in aor_table:
            aor_table["label"] = "unknown"
        aor_whitelist = set(aor_table["aor"])
        aor_meta = {row["aor"]: row for _, row in aor_table.iterrows()}
    else:
        aor_whitelist = None
        aor_meta = {}

    files = discover_files(base_path, aor_whitelist)
    if not files:
        raise RuntimeError("No matching ch2 BCD files found for the selected AORs")

    found_aors = sorted({extract_aor_from_path(f) for f in files})
    if aor_whitelist is not None:
        missing = sorted(aor_whitelist - set(found_aors))
        if missing:
            raise RuntimeError(
                f"Missing requested AORs in local archive: {missing}"
            )
        if len(found_aors) != EXPECTED_AOR_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_AOR_COUNT} AORs for de Wit 4.5 um, "
                f"found {len(found_aors)}"
            )

    per_aor = {}
    file_summaries = []

    # Per-file extraction (Lewis-style)
    for path in files:
        df, summary = process_bcd_file(path)
        aor = summary.aor_id
        df["aor_id"] = aor
        df["visit_label"] = aor_meta.get(aor, {}).get("label", "unknown")
        per_aor.setdefault(aor, []).append(df)
        file_summaries.append(asdict(summary))

    frames = []
    aor_summaries = []

    # Per-AOR aperture optimization (de Wit-style) and finalization
    for aor in sorted(per_aor):
        df = (
            pd.concat(per_aor[aor], ignore_index=True)
            .sort_values("bjd_utc")
            .reset_index(drop=True)
        )
        offset = float(aor_meta[aor]["offset"]) if "offset" in aor_meta.get(aor, {}) else optimize_aperture_offset(df, APERTURE_R_GRID)
        df = finalize_aor_photometry(df, offset)
        df["visit_index"] = np.arange(len(df), dtype=int)
        frames.append(df)

        kept_out = df[df["frame_ok_outlier"]].copy()
        bjd_start = float(df["bjd_utc"].min())
        bjd_end = float(df["bjd_utc"].max())
        duration_hr = 24.0 * (bjd_end - bjd_start)
        flux_scatter_ppm = 1e6 * robust_std(
            kept_out["flux_norm_visit"].to_numpy(float)
            - moving_median(
                kept_out["flux_norm_visit"].to_numpy(float),
                MOVING_MEDIAN_WIDTH,
            )
        )
        aor_summaries.append(
            asdict(
                AORSummary(
                    aor_id=aor,
                    label=str(df["visit_label"].iloc[0]),
                    n_files=int(df["fits_file"].nunique()),
                    n_frames_raw=int(len(df)),
                    n_frames_kept=int(kept_out["frame_ok_outlier"].sum()),
                    bjd_start=bjd_start,
                    bjd_end=bjd_end,
                    duration_hr=duration_hr,
                    beta_median=float(np.nanmedian(df["beta"])),
                    radius_median=float(np.nanmedian(df["aperture_radius"])),
                    x_median=float(np.nanmedian(df["x_cent"])),
                    y_median=float(np.nanmedian(df["y_cent"])),
                    aperture_offset=float(offset),
                    flux_scatter_ppm=float(flux_scatter_ppm),
                )
            )
        )

    # Campaign concatenation
    phot = (
        pd.concat(frames, ignore_index=True)
        .sort_values("bjd_utc")
        .reset_index(drop=True)
    )
    phot["global_index"] = np.arange(len(phot), dtype=int)

    # Lewis-style segment identification and ramp trimming
    phot = assign_time_segments(phot, gap_hours=SEGMENT_GAP_HOURS)

    if TRIM_FIRST_HOUR:
        phot["frame_ok_trim"] = first_hour_keep_mask(phot, trim_hours=TRIM_HOURS)
    else:
        phot["frame_ok_trim"] = True

    # Centroid/beta clipping per AOR+segment
    phot["frame_ok_posbeta"] = make_position_beta_mask(phot)

    # Final combined mask
    phot["frame_ok"] = (
        phot["frame_ok_outlier"].to_numpy(bool)
        & phot["frame_ok_trim"].to_numpy(bool)
        & phot["frame_ok_posbeta"].to_numpy(bool)
    )

    # Campaign-wide normalization from trimmed + outlier-clean frames
    kept = phot[phot["frame_ok"]].copy()
    global_median = np.nanmedian(kept["flux_raw"])
    phot["flux_norm_global"] = (
        phot["flux_raw"] / global_median if global_median > 0 else np.nan
    )

    # Final photometry CSV
    phot_out = phot[
        [
            "global_index",
            "aor_id",
            "visit_label",
            "visit_index",
            "segment_id",
            "fits_file",
            "cube_index",
            "bjd_utc",
            "flux_raw",
            "flux_norm_visit",
            "flux_norm_global",
            "background",
            "x_cent",
            "y_cent",
            "beta",
            "aperture_radius",
            "aperture_offset",
            "n_hotpix_fixed_frame",
            "frame_ok_outlier",
            "frame_ok_trim",
            "frame_ok_posbeta",
            "frame_ok",
        ]
    ].copy()

    phot_out_path = output_dir / "phase1_hatp2b_45um_photometry.csv"
    phot_out.to_csv(phot_out_path, index=False)

    # AOR summary CSV
    aor_summary_df = pd.DataFrame(aor_summaries)
    aor_summary_path = output_dir / "phase1_hatp2b_45um_aor_summary.csv"
    aor_summary_df.to_csv(aor_summary_path, index=False)

    # File summary CSV
    file_summary_df = pd.DataFrame(file_summaries)
    file_summary_path = output_dir / "phase1_hatp2b_45um_file_summary.csv"
    file_summary_df.to_csv(file_summary_path, index=False)

    # Manifest and duration sanity-check
    summed_aor_duration_hr = float(aor_summary_df["duration_hr"].sum())
    timespan_hr = float(
        24.0
        * (
            phot_out["bjd_utc"].max()
            - phot_out["bjd_utc"].min()
        )
    )
    duration_warning = ""
    lo, hi = EXPECTED_TOTAL_HOURS_RANGE
    if not (lo <= summed_aor_duration_hr <= hi):
        duration_warning = (
            f"Summed AOR duration {summed_aor_duration_hr:.3f} hr "
            f"is outside expected de Wit range [{lo}, {hi}] hr"
        )

    manifest = {
        "base_path": str(base_path),
        "n_files": int(len(file_summary_df)),
        "n_frames_total": int(len(phot_out)),
        "n_frames_kept": int(phot_out["frame_ok"].sum()),
        "n_aors": int(phot_out["aor_id"].nunique()),
        "bjd_start": float(phot_out["bjd_utc"].min()),
        "bjd_end": float(phot_out["bjd_utc"].max()),
        "timespan_hr": timespan_hr,
        "summed_aor_duration_hr": summed_aor_duration_hr,
        "expected_duration_range_hr": [lo, hi],
        "duration_warning": duration_warning,
        "output_files": {
            "photometry_csv": str(phot_out_path),
            "aor_summary_csv": str(aor_summary_path),
            "file_summary_csv": str(file_summary_path),
        },
    }
    manifest_path = output_dir / "phase1_hatp2b_45um_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    # Diagnostic plot
    make_phase1_diag_plot(phot_out, output_dir)


def make_phase1_diag_plot(phot: pd.DataFrame, output_dir: Path):
    """
    Four-panel diagnostic figure:

      (a) x centroid vs observation time from first retained point,
      (b) y centroid,
      (c) noise-pixel beta,
      (d) globally normalized relative flux (flux_norm_global),
          with a dashed line at 1.0.

    Uses only frame_ok points, with time in hours relative to the earliest
    retained BJD_UTC. Ramp trimming (first hour per segment) and centroid/beta
    clipping are already encoded in frame_ok, so panel (d) matches the Lewis
    4.5 um behavior when zoomed to the 2011 phase-curve subset and panels
    (a)–(c) are free of repeated extreme outliers.
    """
    kept = phot[phot["frame_ok"]].copy()
    if kept.empty:
        print("No frame_ok points; skipping diagnostic plot.")
        return

    t0 = float(kept["bjd_utc"].min())
    kept["time_hr"] = (kept["bjd_utc"] - t0) * 24.0

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )

    time_hr = kept["time_hr"].to_numpy(float)

    # (a) X centroid
    axes[0].plot(
        time_hr,
        kept["x_cent"].to_numpy(float),
        "k.",
        ms=2.0,
        alpha=0.7,
    )
    axes[0].set_ylabel("X position\n(pixels)")
    axes[0].text(
        0.01,
        0.90,
        "(a)",
        transform=axes[0].transAxes,
        fontweight="bold",
    )

    # (b) Y centroid
    axes[1].plot(
        time_hr,
        kept["y_cent"].to_numpy(float),
        "k.",
        ms=2.0,
        alpha=0.7,
    )
    axes[1].set_ylabel("Y position\n(pixels)")
    axes[1].text(
        0.01,
        0.90,
        "(b)",
        transform=axes[1].transAxes,
        fontweight="bold",
    )

    # (c) Noise-pixel beta
    axes[2].plot(
        time_hr,
        kept["beta"].to_numpy(float),
        "k.",
        ms=2.0,
        alpha=0.7,
    )
    axes[2].set_ylabel(r"Noise pixels ($\beta$)")
    axes[2].text(
        0.01,
        0.90,
        "(c)",
        transform=axes[2].transAxes,
        fontweight="bold",
    )

    # (d) Relative flux (global normalization)
    axes[3].plot(
        time_hr,
        kept["flux_norm_global"].to_numpy(float),
        "k.",
        ms=2.0,
        alpha=0.85,
    )
    axes[3].axhline(
        1.0,
        color="firebrick",
        lw=0.9,
        ls="--",
        alpha=0.75,
    )
    axes[3].set_xlabel(
        "Observation time from first retained point (hr)"
    )
    axes[3].set_ylabel("Relative flux")
    axes[3].text(
        0.01,
        0.90,
        "(d)",
        transform=axes[3].transAxes,
        fontweight="bold",
    )

    for ax in axes:
        ax.grid(alpha=0.15)

    out_path = output_dir / DIAG_FIGURE_NAME
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved diagnostic figure: {out_path}")


# -------------------------------------------------------------------------
# CLI and main
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1 extraction for the de Wit HAT-P-2b 4.5 um dataset"
    )
    p.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Root directory containing {AOR}/r{AOR}/ch2/bcd/*_bcd.fits",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for CSV outputs and diagnostic plot",
    )
    p.add_argument(
        "--aor-table",
        type=Path,
        default=None,
        help="CSV/JSON table with columns aor,label,offset(optional)",
    )
    p.add_argument(
        "--allow-all-aors",
        action="store_true",
        help="Allow processing all discovered AORs when no AOR table is provided",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_phase1_outputs(
        args.base_path,
        args.output_dir,
        args.aor_table,
        allow_all=args.allow_all_aors,
    )


if __name__ == "__main__":
    main()