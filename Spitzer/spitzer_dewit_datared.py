from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit

# TODO: experiment with beta_red

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

DEFAULT_GLOB = "*/r*/ch2/bcd/SPITZER_I2_*_bcd.fits"

FRAME_SHAPE = (32, 32)
CUBE_LEN = 64

SIGMA_HOTPIX = 4.5
SIGMA_OUTLIER = 4.5
MOVING_MEDIAN_WIDTH = 16

BACKGROUND_RMIN = 10.0
CENTROID_R = 3.5

APERTURE_R_GRID = np.round(np.arange(-1.20, 0.20 + 1e-9, 0.05), 2)
APERTURE_R_FLOOR = 1.00
APERTURE_R_CEIL = 3.00

SEGMENT_GAP_HOURS = 1.0
TRIM_FIRST_HOUR = True
TRIM_HOURS = 1.0

EXPECTED_AOR_COUNT = 28

DIAG_FIGURE_NAME = "phase1_hatp2b_45um_diag.png"


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


def compute_bjd_utc_for_cube(header, n_frames: int):
    mjd0 = get_header_value(header, "MBJD_OBS", "BMJD_OBS")
    if mjd0 is None:
        raise KeyError("Missing MBJD_OBS/BMJD_OBS in FITS header")

    aintbeg = get_header_value(header, "AINTBEG")
    atimeend = get_header_value(header, "ATIMEEND")
    framtime = float(get_header_value(header, "FRAMTIME", "EXPTIME", default=0.4))

    if aintbeg is not None and atimeend is not None:
        total = float(atimeend) - float(aintbeg)
        expected = n_frames * framtime
        if (not np.isfinite(total)) or total <= 0:
            total = expected
        dt = total / n_frames
    else:
        dt = framtime

    mids = (np.arange(n_frames, dtype=float) + 0.5) * dt
    jd_mid = float(mjd0) + mids / 86400.0 + 2400000.5
    return jd_mid


def estimate_background(image: np.ndarray, rmin: float = BACKGROUND_RMIN) -> float:
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
    if not np.isfinite(beta) or beta <= 0:
        return np.nan
    r = math.sqrt(beta) + offset
    return float(np.clip(r, APERTURE_R_FLOOR, APERTURE_R_CEIL))


def process_bcd_file(path: Path):
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header

    if data.ndim != 3 or data.shape[0] != CUBE_LEN or data.shape[1:] != FRAME_SHAPE:
        raise ValueError(f"Unexpected cube shape {data.shape} in {path}")

    bjd = compute_bjd_utc_for_cube(header, data.shape[0])
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

    return pd.DataFrame(rows)

# TODO: This function can get computationally optimized.
def optimize_aperture_offset(df: pd.DataFrame, offsets: Iterable[float]) -> float:
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


def assign_time_segments(photometry: pd.DataFrame, gap_hours: float = SEGMENT_GAP_HOURS) -> pd.DataFrame:
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
    keep = np.ones(len(photometry), dtype=bool)
    trim_days = trim_hours / 24.0

    for _, group in photometry.groupby("segment_id", sort=False):
        indices = group.index.to_numpy()
        times = group["bjd_utc"].to_numpy(dtype=float)
        segment_start = np.nanmin(times)
        keep[indices] = times >= segment_start + trim_days

    return keep


def make_position_beta_mask(
    photometry: pd.DataFrame,
    sigma_pos: float = 5.0,
    sigma_beta: float = 5.0,
    hard_dx: float = 0.30,
    hard_dy: float = 0.30,
) -> np.ndarray:
    mask = np.ones(len(photometry), dtype=bool)

    for (_, _), group in photometry.groupby(["aor_id", "segment_id"], sort=False):
        idx = group.index.to_numpy()
        x = group["x_cent"].to_numpy(float)
        y = group["y_cent"].to_numpy(float)
        b = group["beta"].to_numpy(float)

        x0 = np.nanmedian(x)
        y0 = np.nanmedian(y)
        b0 = np.nanmedian(b)

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


def make_phase1_outputs(base_path: Path, output_dir: Path, aor_table_path: Optional[Path], allow_all: bool = False):
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
    for path in files:
        df = process_bcd_file(path)
        aor = extract_aor_from_path(path)
        df["aor_id"] = aor
        df["visit_label"] = aor_meta.get(aor, {}).get("label", "unknown")
        per_aor.setdefault(aor, []).append(df)

    aor_frames = {}
    for aor in per_aor:
        aor_frames[aor] = (
            pd.concat(per_aor[aor], ignore_index=True)
            .sort_values("bjd_utc")
            .reset_index(drop=True)
        )

    visit_order = sorted(aor_frames, key=lambda a: aor_frames[a]["bjd_utc"].min())
    aor_to_visit_index = {aor: i for i, aor in enumerate(visit_order)}

    frames = []
    for aor in sorted(aor_frames):
        df = aor_frames[aor]
        offset = (
            float(aor_meta[aor]["offset"])
            if "offset" in aor_meta.get(aor, {})
            else optimize_aperture_offset(df, APERTURE_R_GRID)
        )
        df = finalize_aor_photometry(df, offset)
        df["visit_index"] = aor_to_visit_index[aor]
        frames.append(df)

    phot = (
        pd.concat(frames, ignore_index=True)
        .sort_values("bjd_utc")
        .reset_index(drop=True)
    )
    phot["global_index"] = np.arange(len(phot), dtype=int)

    phot = assign_time_segments(phot, gap_hours=SEGMENT_GAP_HOURS)

    if TRIM_FIRST_HOUR:
        phot["frame_ok_trim"] = first_hour_keep_mask(phot, trim_hours=TRIM_HOURS)
    else:
        phot["frame_ok_trim"] = True

    phot["frame_ok_posbeta"] = make_position_beta_mask(phot)

    phot["frame_ok"] = (
        phot["frame_ok_outlier"].to_numpy(bool)
        & phot["frame_ok_trim"].to_numpy(bool)
        & phot["frame_ok_posbeta"].to_numpy(bool)
    )

    kept = phot[phot["frame_ok"]]
    global_scale = np.nanmedian(kept["flux_norm_visit"])
    phot["flux_norm_global"] = (
        phot["flux_norm_visit"] / global_scale if global_scale > 0 else np.nan # NOTE: Possible bug here
    )

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
    print(
        f"Saved {phot_out_path} : {len(phot_out)} frames, "
        f"{int(phot_out['frame_ok'].sum())} kept, "
        f"{phot_out['aor_id'].nunique()} AORs"
    )

    make_phase1_diag_plot(phot_out, output_dir)


def make_phase1_diag_plot(phot: pd.DataFrame, output_dir: Path):
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

    axes[0].plot(time_hr, kept["x_cent"].to_numpy(float), "k.", ms=2.0, alpha=0.7)
    axes[0].set_ylabel("X position\n(pixels)")
    axes[0].text(0.01, 0.90, "(a)", transform=axes[0].transAxes, fontweight="bold")

    axes[1].plot(time_hr, kept["y_cent"].to_numpy(float), "k.", ms=2.0, alpha=0.7)
    axes[1].set_ylabel("Y position\n(pixels)")
    axes[1].text(0.01, 0.90, "(b)", transform=axes[1].transAxes, fontweight="bold")

    axes[2].plot(time_hr, kept["beta"].to_numpy(float), "k.", ms=2.0, alpha=0.7)
    axes[2].set_ylabel(r"Noise pixels ($\beta$)")
    axes[2].text(0.01, 0.90, "(c)", transform=axes[2].transAxes, fontweight="bold")

    axes[3].plot(time_hr, kept["flux_norm_global"].to_numpy(float), "k.", ms=2.0, alpha=0.85)
    axes[3].axhline(1.0, color="firebrick", lw=0.9, ls="--", alpha=0.75)
    axes[3].set_xlabel("Observation time from first retained point (hr)")
    axes[3].set_ylabel("Relative flux")
    axes[3].text(0.01, 0.90, "(d)", transform=axes[3].transAxes, fontweight="bold")

    for ax in axes:
        ax.grid(alpha=0.15)

    out_path = output_dir / DIAG_FIGURE_NAME
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved diagnostic figure: {out_path}")


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
        help="Directory for the photometry CSV and diagnostic plot",
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