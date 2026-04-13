import torch
import torch.nn.functional as F

import config


def vae_loss(x_recon, x, mu, logvar, kl_weight=1.0):
    """
    Combined VAE loss: reconstruction (MSE) + KL divergence.

    Returns:
        total_loss, recon_loss, kl_loss (all scalar tensors)
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def vqvae_loss(x_recon, x, codebook_loss, commitment_loss, cls_loss,
               commitment_cost=config.COMMITMENT_COST,
               codebook_cost=config.CODEBOOK_COST,
               cls_weight=1.0):
    """
    VQ-VAE loss: reconstruction + codebook alignment + commitment + classification.

    Returns:
        total_loss, recon_loss, cb_loss (weighted), commit_loss (weighted), cls_loss (weighted)
    """
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    cb_weighted = codebook_cost * codebook_loss
    commit_weighted = commitment_cost * commitment_loss
    cls_weighted = cls_weight * cls_loss
    total_loss = recon_loss + cb_weighted + commit_weighted + cls_weighted
    return total_loss, recon_loss, cb_weighted, commit_weighted, cls_weighted
