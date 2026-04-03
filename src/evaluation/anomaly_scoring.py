import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_anomaly_scores(model, dataset, device, batch_size=512):
    """
    Compute per-recording anomaly scores using multiple methods:
    1. Reconstruction error (MSE)
    2. Latent distance (L2 norm of mu)
    3. ELBO score (recon + KL per sample) — the most principled VAE anomaly score

    Returns:
        dict with keys: subject_ids, study_ids, labels,
                        recon_scores, latent_scores, combined_scores, elbo_scores
    """
    # Use all 10 windows per recording for evaluation
    dataset.windows_per_sample = 10
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model.eval()
    results = {
        "subject_ids": [],
        "study_ids": [],
        "labels": [],
        "recon_scores": [],
        "latent_scores": [],
        "combined_scores": [],
        "elbo_scores": [],
    }

    for batch in loader:
        x = batch["signal"].squeeze(0).to(device)  # (10, 250)
        x_recon, mu, logvar = model(x)

        # Per-window reconstruction error
        recon_per_window = F.mse_loss(x_recon, x, reduction="none").mean(dim=1)  # (10,)

        # Per-window KL divergence: -0.5 * (1 + logvar - mu^2 - exp(logvar)), summed over latent dims
        kl_per_window = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)  # (10,)

        # Per-window ELBO score (negative ELBO = reconstruction + KL)
        elbo_per_window = recon_per_window + kl_per_window

        # Aggregate: 90th percentile for all scores
        recon_score = torch.quantile(recon_per_window, 0.9).item()
        latent_score = torch.quantile(mu.pow(2).sum(dim=1).sqrt(), 0.9).item()
        elbo_score = torch.quantile(elbo_per_window, 0.9).item()

        # Combined: recon + latent
        combined_score = recon_score + 0.1 * latent_score

        results["subject_ids"].append(batch["subject_id"].item())
        results["study_ids"].append(batch["study_id"].item())
        results["labels"].append(batch["label"].item())
        results["recon_scores"].append(recon_score)
        results["latent_scores"].append(latent_score)
        results["combined_scores"].append(combined_score)
        results["elbo_scores"].append(elbo_score)

    # Reset
    dataset.windows_per_sample = 1

    for key in ["recon_scores", "latent_scores", "combined_scores", "elbo_scores", "labels"]:
        results[key] = np.array(results[key])

    return results
