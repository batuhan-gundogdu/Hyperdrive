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
    map_location=device, weights_only=False,
)
base_model.load_state_dict(checkpoint["model_state_dict"])
base_model.eval()

# Per-patient evaluation
all_base_labels, all_base_recon, all_base_latent, all_base_combined, all_base_elbo = [], [], [], [], []
all_ft_labels, all_ft_recon, all_ft_latent, all_ft_combined, all_ft_elbo = [], [], [], [], []
patient_results = []

max_patients = min(200, len(candidates))
for pid in tqdm(candidates[:max_patients], desc="Fine-tuning patients"):
    patient_entries = [e for e in test_entries if e["subject_id"] == pid]

    # Sort by time for temporal split
    patient_entries.sort(key=lambda e: e["ecg_time"])

    # Use first 60% normal recordings for fine-tuning, rest for evaluation
    normal_entries = [e for e in patient_entries if e["is_normal"]]
    n_finetune = max(1, int(len(normal_entries) * 0.6))
    finetune_entries = normal_entries[:n_finetune]
    finetune_study_ids = {e["study_id"] for e in finetune_entries}
    eval_entries = [e for e in patient_entries if e["study_id"] not in finetune_study_ids]

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
    all_base_recon.extend(base_results["recon_scores"])
    all_base_latent.extend(base_results["latent_scores"])
    all_base_combined.extend(base_results["combined_scores"])
    all_base_elbo.extend(base_results["elbo_scores"])
    all_ft_labels.extend(ft_results["labels"])
    all_ft_recon.extend(ft_results["recon_scores"])
    all_ft_latent.extend(ft_results["latent_scores"])
    all_ft_combined.extend(ft_results["combined_scores"])
    all_ft_elbo.extend(ft_results["elbo_scores"])

    # Per-patient metrics (if patient has both classes in eval set)
    if len(set(base_results["labels"])) == 2:
        base_m = compute_metrics(base_results["labels"], base_results["combined_scores"])
        ft_m = compute_metrics(ft_results["labels"], ft_results["combined_scores"])
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
all_ft_labels = np.array(all_ft_labels)

def print_metrics(title, base_scores, ft_scores, base_labels, ft_labels):
    if len(set(base_labels)) < 2:
        print(f"  {title}: Skipped (only one class)")
        return None, None
    base_m = compute_metrics(base_labels, np.array(base_scores))
    ft_m = compute_metrics(ft_labels, np.array(ft_scores))
    print(f"\n{title}:")
    print(f"{'Metric':<20} {'Base':>12} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 60)
    for key in ["auroc", "auprc", "best_f1", "precision", "recall"]:
        delta = ft_m[key] - base_m[key]
        print(f"{key:<20} {base_m[key]:>12.4f} {ft_m[key]:>12.4f} {delta:>+12.4f}")
    return base_m, ft_m

base_recon_m, ft_recon_m = print_metrics("Reconstruction Score",
    all_base_recon, all_ft_recon, all_base_labels, all_ft_labels)
base_latent_m, ft_latent_m = print_metrics("Latent Score",
    all_base_latent, all_ft_latent, all_base_labels, all_ft_labels)
base_combined_m, ft_combined_m = print_metrics("Combined Score (recon + 0.1*latent)",
    all_base_combined, all_ft_combined, all_base_labels, all_ft_labels)
base_elbo_m, ft_elbo_m = print_metrics("ELBO Score (recon + KL per sample)",
    all_base_elbo, all_ft_elbo, all_base_labels, all_ft_labels)

if patient_results:
    improved = sum(1 for r in patient_results if r["ft_auroc"] > r["base_auroc"])
    print(f"\nPatients improved: {improved}/{len(patient_results)} "
          f"({100*improved/len(patient_results):.1f}%)")
    mean_base = np.mean([r["base_auroc"] for r in patient_results])
    mean_ft = np.mean([r["ft_auroc"] for r in patient_results])
    print(f"Mean AUROC (combined): base={mean_base:.4f}, fine-tuned={mean_ft:.4f}")

# Save results
results_data = {
    "base_recon": base_recon_m,
    "ft_recon": ft_recon_m,
    "base_latent": base_latent_m,
    "ft_latent": ft_latent_m,
    "base_combined": base_combined_m,
    "ft_combined": ft_combined_m,
    "base_elbo": base_elbo_m,
    "ft_elbo": ft_elbo_m,
    "patient_results": patient_results,
    "n_candidates": len(candidates),
    "n_evaluated": max_patients,
}
results_path = os.path.join(config.CHECKPOINT_DIR, "eval_results.pkl")
with open(results_path, "wb") as f:
    pickle.dump(results_data, f)
print(f"\nResults saved to {results_path}")
