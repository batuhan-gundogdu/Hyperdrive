import torch
import torch.nn.functional as F


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
