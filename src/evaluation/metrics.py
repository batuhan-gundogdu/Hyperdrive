import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score,
)


def compute_metrics(labels, scores):
    """Compute AUROC, AUPRC, and best-F1 metrics."""
    auroc = roc_auc_score(labels, scores)

    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
    auprc = auc(recall_curve, precision_curve)

    # Find threshold that maximizes F1
    f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    preds = (scores >= best_threshold).astype(int)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": f1_scores[best_idx],
        "best_threshold": best_threshold,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def compare_base_vs_finetuned(base_results, finetuned_results):
    """Compare base model vs fine-tuned model on the same patients."""
    base_metrics = compute_metrics(base_results["labels"], base_results["recon_scores"])
    ft_metrics = compute_metrics(finetuned_results["labels"], finetuned_results["recon_scores"])

    print("=" * 60)
    print(f"{'Metric':<20} {'Base':>12} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 60)
    for key in ["auroc", "auprc", "best_f1", "precision", "recall"]:
        delta = ft_metrics[key] - base_metrics[key]
        print(f"{key:<20} {base_metrics[key]:>12.4f} {ft_metrics[key]:>12.4f} {delta:>+12.4f}")
    print("=" * 60)

    return base_metrics, ft_metrics
