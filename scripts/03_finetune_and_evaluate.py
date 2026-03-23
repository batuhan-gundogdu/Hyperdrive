"""Fine-tune per patient and compare base vs fine-tuned performance."""
import os
import sys
import pickle
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from tqdm import tqdm

import config
from src.models.vae import VAE
from src.data.dataset import ECGWindowDataset, PatientDataset
from src.data.splits import split_by_patient, get_finetune_candidates
from src.training.finetune_patient import finetune_for_patient
from src.evaluation.anomaly_scoring import compute_anomaly_scores
from src.evaluation.metrics import compute_metrics

# Load cache
cache_path = os.path.join(config.CACHE_DIR, "dataset_cache.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

entries = cache["entries"]
global_mean = cache["global_mean"]
global_std = cache["global_std"]

_, _, test_entries = split_by_patient(entries)
candidates = get_finetune_candidates(test_entries)
print(f"Fine-tune candidates (mixed patients with >={config.MIN_RECORDINGS_FOR_FINETUNE} recordings): {len(candidates)}")

# Load base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = VAE().to(device)
checkpoint = torch.load(
    os.path.join(config.CHECKPOINT_DIR, "base_model.pt"),
    map_location=device, weights_only=True,
)
base_model.load_state_dict(checkpoint["model_state_dict"])
base_model.eval()

# Per-patient evaluation
all_base_labels, all_base_scores = [], []
all_ft_labels, all_ft_scores = [], []
patient_results = []

for pid in tqdm(candidates[:100], desc="Fine-tuning patients"):
    patient_entries = [e for e in test_entries if e["subject_id"] == pid]

    # Sort by time for temporal split
    patient_entries.sort(key=lambda e: e["ecg_time"])

    # Use first 60% normal recordings for fine-tuning, rest for evaluation
    normal_entries = [e for e in patient_entries if e["is_normal"]]
    n_finetune = max(1, int(len(normal_entries) * 0.6))
    finetune_entries = normal_entries[:n_finetune]
    eval_entries = [e for e in patient_entries if e not in finetune_entries]

    if len(eval_entries) == 0:
        continue

    # Fine-tune
    ft_dataset = ECGWindowDataset(finetune_entries, global_mean, global_std)
    ft_model = finetune_for_patient(base_model, ft_dataset, device)

    # Evaluate both models on held-out recordings
    eval_dataset = ECGWindowDataset(eval_entries, global_mean, global_std)
    base_results = compute_anomaly_scores(base_model, eval_dataset, device)
    ft_results = compute_anomaly_scores(ft_model, eval_dataset, device)

    all_base_labels.extend(base_results["labels"])
    all_base_scores.extend(base_results["recon_scores"])
    all_ft_labels.extend(ft_results["labels"])
    all_ft_scores.extend(ft_results["recon_scores"])

    # Per-patient metrics (if patient has both classes in eval set)
    if len(set(base_results["labels"])) == 2:
        base_m = compute_metrics(base_results["labels"], base_results["recon_scores"])
        ft_m = compute_metrics(ft_results["labels"], ft_results["recon_scores"])
        patient_results.append({
            "subject_id": pid,
            "base_auroc": base_m["auroc"],
            "ft_auroc": ft_m["auroc"],
            "base_recall": base_m["recall"],
            "ft_recall": ft_m["recall"],
            "base_precision": base_m["precision"],
            "ft_precision": ft_m["precision"],
        })

# Aggregate results
print("\n" + "=" * 70)
print("AGGREGATE RESULTS (across all evaluated patients)")
print("=" * 70)

all_base_labels = np.array(all_base_labels)
all_base_scores = np.array(all_base_scores)
all_ft_labels = np.array(all_ft_labels)
all_ft_scores = np.array(all_ft_scores)

if len(set(all_base_labels)) == 2:
    base_agg = compute_metrics(all_base_labels, all_base_scores)
    ft_agg = compute_metrics(all_ft_labels, all_ft_scores)

    print(f"{'Metric':<20} {'Base':>12} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 60)
    for key in ["auroc", "auprc", "best_f1", "precision", "recall"]:
        delta = ft_agg[key] - base_agg[key]
        print(f"{key:<20} {base_agg[key]:>12.4f} {ft_agg[key]:>12.4f} {delta:>+12.4f}")

if patient_results:
    improved = sum(1 for r in patient_results if r["ft_auroc"] > r["base_auroc"])
    print(f"\nPatients improved: {improved}/{len(patient_results)} "
          f"({100*improved/len(patient_results):.1f}%)")
    mean_base = np.mean([r["base_auroc"] for r in patient_results])
    mean_ft = np.mean([r["ft_auroc"] for r in patient_results])
    print(f"Mean AUROC: base={mean_base:.4f}, fine-tuned={mean_ft:.4f}")
