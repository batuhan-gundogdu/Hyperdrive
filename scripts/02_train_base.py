"""Train the base VAE model on the full training set."""
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

train_dataset = ECGWindowDataset(train_entries, global_mean, global_std)
val_dataset = ECGWindowDataset(val_entries, global_mean, global_std)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VAE().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")

model = train_base_model(model, train_dataset, val_dataset, device)
print("Base model training complete.")
