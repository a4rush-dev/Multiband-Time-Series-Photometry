import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_lightcurve(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    t = np.array(df["time"])
    f = np.array(df["flux"])
    return t, f


def normalize_flux(f: np.ndarray) -> np.ndarray:
    # Normalize
    median_flux = np.median(f)
    f_norm = f / median_flux
    return f_norm


def clean_bad_data(t: np.ndarray, f_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Clean bad data
    mask_good = np.isfinite(t) & np.isfinite(f_norm)
    t = t[mask_good]
    f_norm = f_norm[mask_good]
    return t, f_norm


def inject_signals(
    t: np.ndarray,
    f_norm: np.ndarray,
    period: float,
) -> tuple[np.ndarray, float, float]:
    orbital_freq = 1.0 / period

    # Injection parameters
    f_79 = 79 * orbital_freq  # 79th harmonic, can be changed for testing/diagnossi
    f_91 = 91 * orbital_freq  # 91st harmonic
    A_79 = 0.001  # amplitude for 79f
    A_91 = 0.001  # amplitude for 91f
    phi_79 = np.random.uniform(0, 2*np.pi)
    phi_91 = np.random.uniform(0, 2*np.pi)

    print(f"\nInjecting signals:")
    print(f"  79f_orb = {f_79:.3f} day^-1 (P = {1/f_79:.3f} d = {24/f_79:.2f} hr)")
    print(f"  91f_orb = {f_91:.3f} day^-1 (P = {1/f_91:.3f} d = {24/f_91:.2f} hr)")

    signal_79 = A_79 * np.sin(2*np.pi * f_79 * t + phi_79)
    signal_91 = A_91 * np.sin(2*np.pi * f_91 * t + phi_91)

    f_injected = f_norm + signal_79 + signal_91
    return f_injected, f_79, f_91


def find_best_t0(
    t: np.ndarray,
    f_norm: np.ndarray,
    period: float,
) -> float:
    ntrial = 1000
    trial_t0 = np.linspace(t[0], t[0] + period, ntrial)
    depths = []

    for t0_test in trial_t0:
        phase = ((t - t0_test) / period) % 1.0
        phase = np.where(phase > 0.5, phase - 1, phase)
        in_transit = np.abs(phase) < 0.02
        if np.sum(in_transit) > 10:
            depths.append(np.median(f_norm[in_transit]))
        else:
            depths.append(1.0)

    t0_best = trial_t0[np.argmin(depths)]
    print(f"\nBest t0 for folding: {t0_best:.6f}")
    return t0_best


def compute_phase(
    t: np.ndarray,
    t0_best: float,
    period: float,
) -> np.ndarray:
    phase_raw = ((t - t0_best) / period) % 1.0
    phase_raw = np.where(phase_raw > 0.5, phase_raw - 1, phase_raw)
    return phase_raw


def sort_by_phase(phase_raw: np.ndarray) -> np.ndarray:
    # Sort by phase
    sort_idx = np.argsort(phase_raw)
    return sort_idx


def make_time_domain_plots(
    t: np.ndarray,
    f_norm: np.ndarray,
    f_injected: np.ndarray,
    f_79: float,
    f_91: float,
) -> None:
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Show only first 50 days
    mask_zoom = (t < t[0] + 50)

    # Raw data
    ax1.plot(t[mask_zoom], f_norm[mask_zoom], 'k.', ms=1, alpha=0.5)
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title('Raw TESS Data (first 50 days)')
    ax1.grid(alpha=0.3)

    ax2.plot(t[mask_zoom], f_injected[mask_zoom], 'b.', ms=1, alpha=0.5)
    ax2.set_xlabel('Time (BJD)')
    ax2.set_ylabel('Normalized Flux')
    ax2.set_title(
        f'Injected Data (79f + 91f added)\n'
        f'79f period={24/f_79:.2f}hr, 91f period={24/f_91:.2f}hr'
    )
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def make_phase_fold_plots(
    phase_raw: np.ndarray,
    sort_idx: np.ndarray,
    f_norm: np.ndarray,
    f_injected: np.ndarray,
) -> None:
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Raw folded
    ax3.plot(phase_raw[sort_idx], f_norm[sort_idx], 'k.', ms=1, alpha=0.3)
    ax3.set_ylabel('Normalized Flux')
    ax3.set_title('Phase-Folded Raw TESS Data')
    ax3.grid(alpha=0.3)
    ax3.set_xlim(-0.5, 0.5)
    ax3.invert_yaxis()

    # Injected folded
    ax4.plot(phase_raw[sort_idx], f_injected[sort_idx], 'b.', ms=1, alpha=0.3)
    ax4.set_xlabel('Phase')
    ax4.set_ylabel('Normalized Flux')
    ax4.set_title('Phase-Folded Injected Data (79f + 91f)')
    ax4.grid(alpha=0.3)
    ax4.set_xlim(-0.5, 0.5)
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.show()


def main() -> None:
    t, f = load_lightcurve("hatp2_tess_lightcurve.csv")
    f_norm = normalize_flux(f)
    t, f_norm = clean_bad_data(t, f_norm)

    print(f"TESS data: {len(t)} points over {t.max() - t.min():.1f} days")

    period = 5.6334729

    f_injected, f_79, f_91 = inject_signals(t, f_norm, period)
    t0_best = find_best_t0(t, f_norm, period)

    phase_raw = compute_phase(t, t0_best, period)
    sort_idx = sort_by_phase(phase_raw)

    make_time_domain_plots(t, f_norm, f_injected, f_79, f_91)
    make_phase_fold_plots(phase_raw, sort_idx, f_norm, f_injected)


if __name__ == "__main__":
    main()