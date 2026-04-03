"""Train the base VAE model on normal ECGs only (for anomaly detection)."""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import config
from src.models.vae import VAE
from src.data.dataset import ECGWindowDataset
from src.data.splits import split_by_patient
from src.training.train_base import train_base_model

# Load cache
cache_path = os.path.join(config.CACHE_DIR, "dataset_cache.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

entries = cache["entries"]
global_mean = cache["global_mean"]
global_std = cache["global_std"]

print("Splitting data by patient...")
train_entries, val_entries, test_entries = split_by_patient(entries)
print(f"  Train: {len(train_entries)} recordings")
print(f"  Val:   {len(val_entries)} recordings")
print(f"  Test:  {len(test_entries)} recordings")

# For anomaly detection: train ONLY on normal ECGs
# Abnormal ECGs should then have higher reconstruction error
train_normal = [e for e in train_entries if e["is_normal"]]
val_normal = [e for e in val_entries if e["is_normal"]]
print(f"  Train normal: {len(train_normal)} ({100*len(train_normal)/len(train_entries):.1f}%)")
print(f"  Val normal:   {len(val_normal)} ({100*len(val_normal)/len(val_entries):.1f}%)")

train_dataset = ECGWindowDataset(train_normal, global_mean, global_std)
val_dataset = ECGWindowDataset(val_normal, global_mean, global_std)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VAE().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")

model, history = train_base_model(model, train_dataset, val_dataset, device)
print("Base model training complete.")

# Save training curves
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
epochs_range = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs_range, history["train_loss"], label="Train")
axes[0].plot(epochs_range, history["val_loss"], label="Val")
axes[0].set_title("Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_range, history["train_recon"], label="Train")
axes[1].plot(epochs_range, history["val_recon"], label="Val")
axes[1].set_title("Reconstruction Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

axes[2].plot(epochs_range, history["train_kl"], label="Train KL")
axes[2].plot(epochs_range, history["val_kl"], label="Val KL")
axes[2].plot(epochs_range, history["kl_weight"], label="KL Weight", linestyle="--")
axes[2].set_title("KL Divergence & Weight")
axes[2].set_xlabel("Epoch")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(config.CHECKPOINT_DIR, "training_curves.png"), dpi=150)
print("Saved training curves.")
