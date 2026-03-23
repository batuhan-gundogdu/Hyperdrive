import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.models.loss import vae_loss


def train_epoch(model, dataloader, optimizer, kl_weight, device):
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        x = batch["signal"].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_recon, x, mu, logvar, kl_weight)
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum += kl.item()
        n_batches += 1

    return {
        "total": total_loss_sum / n_batches,
        "recon": recon_loss_sum / n_batches,
        "kl": kl_loss_sum / n_batches,
    }


@torch.no_grad()
def eval_epoch(model, dataloader, kl_weight, device):
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["signal"].to(device)
        x_recon, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_recon, x, mu, logvar, kl_weight)

        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        kl_loss_sum += kl.item()
        n_batches += 1

    return {
        "total": total_loss_sum / n_batches,
        "recon": recon_loss_sum / n_batches,
        "kl": kl_loss_sum / n_batches,
    }


def train_base_model(model, train_dataset, val_dataset, device, epochs=config.EPOCHS_BASE,
                     lr=config.LR_BASE, kl_weight=config.KL_WEIGHT,
                     batch_size=config.BATCH_SIZE, patience=config.PATIENCE):
    """Train the base VAE model with early stopping."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(config.CHECKPOINT_DIR, "base_model.pt")

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, kl_weight, device)
        val_metrics = eval_epoch(model, val_loader, kl_weight, device)
        scheduler.step(val_metrics["total"])

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs} | "
              f"Train: {train_metrics['total']:.6f} (recon={train_metrics['recon']:.6f}, kl={train_metrics['kl']:.6f}) | "
              f"Val: {val_metrics['total']:.6f} (recon={val_metrics['recon']:.6f}, kl={val_metrics['kl']:.6f}) | "
              f"LR: {current_lr:.2e}")

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, best_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    return model
