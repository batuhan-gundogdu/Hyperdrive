import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_anomaly_scores(model, dataset, device, batch_size=512):
    """
    Compute per-recording anomaly scores using reconstruction error and latent distance.

    Returns:
        dict with keys: subject_ids, study_ids, labels,
                        recon_scores, latent_scores, combined_scores
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
    }

    for batch in loader:
        x = batch["signal"].squeeze(0).to(device)  # (10, 250)
        x_recon, mu, logvar = model(x)

        # Per-window reconstruction error, then max across windows
        recon_error = F.mse_loss(x_recon, x, reduction="none").mean(dim=1)  # (10,)
        recon_score = recon_error.max().item()

        # Latent distance: L2 norm of mean encoding
        latent_score = mu.pow(2).sum(dim=1).sqrt().max().item()

        results["subject_ids"].append(batch["subject_id"].item())
        results["study_ids"].append(batch["study_id"].item())
        results["labels"].append(batch["label"].item())
        results["recon_scores"].append(recon_score)
        results["latent_scores"].append(latent_score)

    # Reset
    dataset.windows_per_sample = 1

    for key in ["recon_scores", "latent_scores", "labels"]:
        results[key] = np.array(results[key])

    return results
