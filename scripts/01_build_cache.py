"""Pre-compute labels, record list, and global signal statistics. Run once."""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.data.labeling import build_record_list
from src.data.preprocessing import compute_global_stats

os.makedirs(config.CACHE_DIR, exist_ok=True)

print("Building record list with labels...")
entries = build_record_list()
print(f"  Total entries: {len(entries)}")

n_normal = sum(1 for e in entries if e["is_normal"])
n_abnormal = len(entries) - n_normal
print(f"  Normal: {n_normal} ({100*n_normal/len(entries):.1f}%)")
print(f"  Abnormal: {n_abnormal} ({100*n_abnormal/len(entries):.1f}%)")

print("Computing global signal statistics (sampling 10K recordings)...")
dat_paths = [e["dat_path"] for e in entries]
global_mean, global_std = compute_global_stats(dat_paths)
print(f"  Global mean: {global_mean:.6f}")
print(f"  Global std:  {global_std:.6f}")

cache = {
    "entries": entries,
    "global_mean": global_mean,
    "global_std": global_std,
}
cache_path = os.path.join(config.CACHE_DIR, "dataset_cache.pkl")
with open(cache_path, "wb") as f:
    pickle.dump(cache, f)
print(f"Saved cache to {cache_path}")
