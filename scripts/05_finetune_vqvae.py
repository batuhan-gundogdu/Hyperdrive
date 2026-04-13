"""Fine-tune VQ-VAE per patient and compare base vs fine-tuned performance."""
import os
import sys
import pickle
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from tqdm import tqdm

import config
from src.models.vqvae import VQVAE
from src.data.dataset import ECGWindowDataset
from src.data.splits import split_by_patient, get_finetune_candidates
from src.training.finetune_patient import finetune_vqvae_for_patient
from src.evaluation.anomaly_scoring import compute_vqvae_anomaly_scores
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
base_model = VQVAE().to(device)
checkpoint = torch.load(
    os.path.join(config.CHECKPOINT_DIR, "vqvae_base_model.pt"),
    map_location=device, weights_only=False,
)
base_model.load_state_dict(checkpoint["model_state_dict"])
base_model.eval()

# Per-patient evaluation
score_keys = ["recon_scores", "codebook_scores", "combined_scores", "code_abnormal_frac"]
all_base = {k: [] for k in score_keys + ["labels"]}
all_ft = {k: [] for k in score_keys + ["labels"]}
patient_results = []

max_patients = min(200, len(candidates))
for pid in tqdm(candidates[:max_patients], desc="Fine-tuning patients"):
    patient_entries = [e for e in test_entries if e["subject_id"] == pid]
    patient_entries.sort(key=lambda e: e["ecg_time"])

    # Use first 60% of ALL recordings for fine-tuning (label-agnostic),
    # rest for evaluation
    n_finetune = max(1, int(len(patient_entries) * 0.6))
    finetune_entries = patient_entries[:n_finetune]
    finetune_study_ids = {e["study_id"] for e in finetune_entries}
    eval_entries = [e for e in patient_entries if e["study_id"] not in finetune_study_ids]

    if len(eval_entries) == 0:
        continue

    # Check eval set has both classes
    eval_labels = set(int(not e["is_normal"]) for e in eval_entries)
    if len(eval_labels) < 2:
        continue

    # Fine-tune on ALL patient recordings (label-agnostic)
    ft_dataset = ECGWindowDataset(finetune_entries, global_mean, global_std)
    ft_model = finetune_vqvae_for_patient(base_model, ft_dataset, device)

    # Evaluate both models
    eval_dataset = ECGWindowDataset(eval_entries, global_mean, global_std)
    base_results = compute_vqvae_anomaly_scores(base_model, eval_dataset, device)
    ft_results = compute_vqvae_anomaly_scores(ft_model, eval_dataset, device)

    for k in score_keys + ["labels"]:
        all_base[k].extend(base_results[k])
        all_ft[k].extend(ft_results[k])

    # Per-patient metrics
    base_m = compute_metrics(base_results["labels"], base_results["combined_scores"])
    ft_m = compute_metrics(ft_results["labels"], ft_results["combined_scores"])
    patient_results.append({
        "subject_id": pid,
        "base_auroc": base_m["auroc"],
        "ft_auroc": ft_m["auroc"],
    })

# Aggregate results
print("\n" + "=" * 70)
print("VQ-VAE AGGREGATE RESULTS (across all evaluated patients)")
print("=" * 70)

for k in score_keys + ["labels"]:
    all_base[k] = np.array(all_base[k])
    all_ft[k] = np.array(all_ft[k])


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
    all_base["recon_scores"], all_ft["recon_scores"], all_base["labels"], all_ft["labels"])
base_cb_m, ft_cb_m = print_metrics("Codebook Distance Score",
    all_base["codebook_scores"], all_ft["codebook_scores"], all_base["labels"], all_ft["labels"])
base_combined_m, ft_combined_m = print_metrics("Combined Score (recon + 0.5*codebook)",
    all_base["combined_scores"], all_ft["combined_scores"], all_base["labels"], all_ft["labels"])

# Code assignment stats
print(f"\nCode Assignment (fraction of windows mapped to abnormal codes):")
abnormal_mask = all_base["labels"] == 1
normal_mask = all_base["labels"] == 0
print(f"  Base model:  normal={all_base['code_abnormal_frac'][normal_mask].mean():.3f}, "
      f"abnormal={all_base['code_abnormal_frac'][abnormal_mask].mean():.3f}")
print(f"  Fine-tuned:  normal={all_ft['code_abnormal_frac'][normal_mask].mean():.3f}, "
      f"abnormal={all_ft['code_abnormal_frac'][abnormal_mask].mean():.3f}")

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
    "base_codebook": base_cb_m,
    "ft_codebook": ft_cb_m,
    "base_combined": base_combined_m,
    "ft_combined": ft_combined_m,
    "patient_results": patient_results,
    "n_candidates": len(candidates),
    "n_evaluated": len(patient_results),
    "code_stats": {
        "base_normal_frac": all_base["code_abnormal_frac"][normal_mask].mean(),
        "base_abnormal_frac": all_base["code_abnormal_frac"][abnormal_mask].mean(),
        "ft_normal_frac": all_ft["code_abnormal_frac"][normal_mask].mean(),
        "ft_abnormal_frac": all_ft["code_abnormal_frac"][abnormal_mask].mean(),
    },
}
results_path = os.path.join(config.CHECKPOINT_DIR, "vqvae_eval_results.pkl")
with open(results_path, "wb") as f:
    pickle.dump(results_data, f)
print(f"\nResults saved to {results_path}")
