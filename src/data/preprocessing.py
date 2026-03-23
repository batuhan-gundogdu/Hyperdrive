import numpy as np
import config


def load_lead_ii_windows(dat_path):
    """Load a .dat file and return Lead II as 10 x 250 windows (downsampled to 250Hz)."""
    raw = np.fromfile(dat_path, dtype=np.int16)
    raw = raw.reshape(-1, config.NUM_LEADS)
    lead_ii = raw[:, config.LEAD_INDEX].astype(np.float32) / config.ADC_GAIN
    # Downsample 500Hz -> 250Hz by taking every other sample
    lead_ii = lead_ii[::2]
    # Reshape into 10 windows of 250 samples
    windows = lead_ii.reshape(config.WINDOWS_PER_RECORDING, config.WINDOW_SIZE)
    return windows


def compute_global_stats(dat_paths, n_samples=10000, seed=42):
    """Compute global mean and std from a random subset of recordings."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dat_paths), size=min(n_samples, len(dat_paths)), replace=False)
    all_values = []
    for idx in indices:
        try:
            windows = load_lead_ii_windows(dat_paths[idx])
            all_values.append(windows.ravel())
        except Exception:
            continue
    all_values = np.concatenate(all_values)
    return float(np.mean(all_values)), float(np.std(all_values))
