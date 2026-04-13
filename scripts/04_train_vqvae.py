"""Train the VQ-VAE base model on ALL ECGs with class-conditional codebook routing."""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import config
from src.models.vqvae import VQVAE
from src.data.dataset import ECGWindowDataset
from src.data.splits import split_by_patient
from src.training.train_vqvae import train_vqvae_base

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

# VQ-VAE trains on ALL data (normal + abnormal) with class-conditional routing
train_normal = sum(1 for e in train_entries if e["is_normal"])
train_abnormal = len(train_entries) - train_normal
print(f"  Train normal: {train_normal} ({100*train_normal/len(train_entries):.1f}%)")
print(f"  Train abnormal: {train_abnormal} ({100*train_abnormal/len(train_entries):.1f}%)")

train_dataset = ECGWindowDataset(train_entries, global_mean, global_std)
val_dataset = ECGWindowDataset(val_entries, global_mean, global_std)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VQVAE().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")
print(f"Codebook: {config.NUM_CODES} codes x {config.EMBED_DIM} dims")
print(f"Normal codes: {config.NORMAL_CODE_IDS}")

model, history = train_vqvae_base(model, train_dataset, val_dataset, device,
                                  val_entries=val_entries, global_mean=global_mean,
                                  global_std=global_std, ft_sim_every=2)
print("VQ-VAE base model training complete.")

# Save training curves
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs_range = range(1, len(history["train_loss"]) + 1)

axes[0, 0].plot(epochs_range, history["train_recon"], label="Train")
axes[0, 0].plot(epochs_range, history["val_recon"], label="Val")
axes[0, 0].set_title("Reconstruction Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].legend()

axes[0, 1].plot(epochs_range, history["train_cls"], label="Cls Loss")
axes[0, 1].plot(epochs_range, history["cls_weight"], label="Cls Weight", linestyle="--")
axes[0, 1].set_title("Classification Loss & Weight")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].legend()

axes[1, 0].plot(epochs_range, history["train_commitment"], label="Commit")
axes[1, 0].plot(epochs_range, history["train_codebook"], label="Codebook")
axes[1, 0].set_title("VQ Losses")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].legend()

axes[1, 1].plot(epochs_range, history["train_routing_acc"], label="Train")
axes[1, 1].plot(epochs_range, history["val_routing_acc"], label="Val (unmasked)")
axes[1, 1].set_title("Code Routing Accuracy")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(config.CHECKPOINT_DIR, "vqvae_training_curves.png"), dpi=150)
print("Saved training curves.")
