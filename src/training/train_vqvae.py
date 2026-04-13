"""Training loop for VQ-VAE with class-conditional codebook routing."""

import os
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.models.loss import vqvae_loss


def train_epoch(model, dataloader, optimizer, device, cls_weight=1.0):
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    cb_loss_sum = 0.0
    commit_loss_sum = 0.0
    cls_loss_sum = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        x = batch["signal"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        x_recon, z_e, z_q, indices, cb_loss, commit_loss, cls_loss = model(x, labels=labels)
        loss, recon, cb_w, commit_w, cls_w = vqvae_loss(
            x_recon, x, cb_loss, commit_loss, cls_loss, cls_weight=cls_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        predicted = model.vq.get_code_type(indices)
        correct += (predicted == labels).sum().item()
        total += labels.shape[0]

        total_loss_sum += loss.item()
        recon_loss_sum += recon.item()
        cb_loss_sum += cb_w.item()
        commit_loss_sum += commit_w.item()
        cls_loss_sum += cls_w.item()
        n_batches += 1

    return {
        "total": total_loss_sum / n_batches,
        "recon": recon_loss_sum / n_batches,
        "codebook": cb_loss_sum / n_batches,
        "commitment": commit_loss_sum / n_batches,
        "cls": cls_loss_sum / n_batches,
        "routing_acc": correct / total if total > 0 else 0.0,
    }


@torch.no_grad()
def eval_epoch(model, dataloader, device):
    model.eval()
    recon_loss_sum = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for batch in dataloader:
        x = batch["signal"].to(device)
        labels = batch["label"].to(device)

        # Evaluate WITHOUT labels (unmasked routing)
        x_recon, _, _, indices, _, _, _ = model(x, labels=None)
        recon = torch.nn.functional.mse_loss(x_recon, x, reduction="mean")

        predicted = model.vq.get_code_type(indices)
        correct += (predicted == labels).sum().item()
        total += labels.shape[0]

        recon_loss_sum += recon.item()
        n_batches += 1

    return {
        "recon": recon_loss_sum / n_batches,
        "routing_acc": correct / total if total > 0 else 0.0,
    }


def simulate_finetune_on_val(model, val_entries, global_mean, global_std, device,
                             n_patients=30):
    """Clone the model, fine-tune per-patient using ONE known-normal recording.

    Protocol: For each patient with both normal and abnormal recordings,
    take the earliest normal recording and fine-tune on it with full loss
    (recon + commit + cls forcing to normal code). Test on remaining recordings.

    Returns dict mapping (epochs, lr) -> metrics dict with accuracy, precision,
    recall, base_accuracy.
    """
    from src.data.dataset import ECGWindowDataset
    from src.training.finetune_patient import finetune_vqvae_for_patient

    # Find val patients with both classes and enough recordings
    patient_data = defaultdict(list)
    for e in val_entries:
        patient_data[e["subject_id"]].append(e)

    candidates = []
    for pid, entries in patient_data.items():
        has_normal = any(e["is_normal"] for e in entries)
        has_abnormal = any(not e["is_normal"] for e in entries)
        if len(entries) >= 3 and has_normal and has_abnormal:
            candidates.append(pid)
    candidates = sorted(candidates)[:n_patients]

    if not candidates:
        return {}

    # Prepare splits: take the earliest normal recording for fine-tuning,
    # everything else for evaluation
    patient_splits = []
    for pid in candidates:
        p_entries = sorted(patient_data[pid], key=lambda e: e["ecg_time"])
        normal_entries = [e for e in p_entries if e["is_normal"]]
        if not normal_entries:
            continue
        # Use just the earliest normal recording
        ft_entries = [normal_entries[0]]
        ft_study_id = normal_entries[0]["study_id"]
        eval_entries = [e for e in p_entries if e["study_id"] != ft_study_id]
        if not eval_entries:
            continue
        # Need both classes in eval set for meaningful precision/recall
        eval_labels = set(int(not e["is_normal"]) for e in eval_entries)
        if len(eval_labels) < 2:
            continue
        patient_splits.append((pid, ft_entries, eval_entries))

    # Settings to sweep: (epochs, lr) — short fine-tuning only
    ft_settings = [
        (1, 1e-4),
        (2, 1e-4),
        (3, 1e-4),
        (5, 1e-4),
    ]

    def compute_metrics(preds, labels):
        """Return dict with accuracy, precision, recall (positive=abnormal)."""
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
        n = len(preds)
        acc = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {"acc": acc, "precision": prec, "recall": rec}

    # Evaluate base model once
    base_preds, base_labels = [], []
    for pid, ft_entries, eval_entries in patient_splits:
        eval_dataset = ECGWindowDataset(eval_entries, global_mean, global_std)
        eval_dataset.windows_per_sample = 10
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            model.eval()
            for batch in eval_loader:
                x = batch["signal"].squeeze(0).to(device)
                label = batch["label"].item()
                _, base_idx = model.reconstruct(x)
                base_pred = model.vq.get_code_type(base_idx)
                base_majority = (base_pred.sum() > 5).long().item()
                base_preds.append(base_majority)
                base_labels.append(label)
        eval_dataset.windows_per_sample = 1

    base_metrics = compute_metrics(base_preds, base_labels)

    # Sweep fine-tuning settings
    results = {}
    for ft_epochs, ft_lr in ft_settings:
        ft_preds, ft_labels = [], []
        for pid, ft_entries, eval_entries in patient_splits:
            ft_dataset = ECGWindowDataset(ft_entries, global_mean, global_std)
            ft_model = finetune_vqvae_for_patient(
                model, ft_dataset, device, epochs=ft_epochs, lr=ft_lr)

            eval_dataset = ECGWindowDataset(eval_entries, global_mean, global_std)
            eval_dataset.windows_per_sample = 10
            eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
            with torch.no_grad():
                ft_model.eval()
                for batch in eval_loader:
                    x = batch["signal"].squeeze(0).to(device)
                    label = batch["label"].item()
                    _, ft_idx = ft_model.reconstruct(x)
                    ft_pred = ft_model.vq.get_code_type(ft_idx)
                    ft_majority = (ft_pred.sum() > 5).long().item()
                    ft_preds.append(ft_majority)
                    ft_labels.append(label)
            eval_dataset.windows_per_sample = 1
            del ft_model

        ft_metrics = compute_metrics(ft_preds, ft_labels)
        results[(ft_epochs, ft_lr)] = {"base": base_metrics, "ft": ft_metrics}

    return results


def train_vqvae_base(model, train_dataset, val_dataset, device,
                     val_entries=None, global_mean=0.0, global_std=1.0,
                     epochs=config.EPOCHS_BASE, lr=config.LR_BASE,
                     batch_size=config.BATCH_SIZE, patience=config.PATIENCE,
                     cls_weight_max=config.CLS_WEIGHT,
                     cls_anneal_epochs=config.CLS_ANNEAL_EPOCHS,
                     ft_sim_every=10):
    """Train the VQ-VAE base model with class-conditional routing.

    - Cls loss is annealed from 0 to cls_weight_max over cls_anneal_epochs.
    - Early stopping monitors val recon loss (not total loss).
    - Fine-tuning simulation sweeps multiple settings every ft_sim_every epochs.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_recon = float("inf")
    epochs_without_improvement = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(config.CHECKPOINT_DIR, "vqvae_base_model.pt")

    history = {
        "train_loss": [], "train_recon": [], "train_cls": [],
        "train_codebook": [], "train_commitment": [],
        "train_routing_acc": [],
        "val_recon": [], "val_routing_acc": [],
        "cls_weight": [], "lr": [],
        "ft_sim": [],  # list of (epoch, results_dict) tuples
    }

    for epoch in range(1, epochs + 1):
        # Anneal classification loss weight
        cls_weight = min(1.0, epoch / cls_anneal_epochs) * cls_weight_max

        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                    cls_weight=cls_weight)
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_cls"].append(train_metrics["cls"])
        history["train_codebook"].append(train_metrics["codebook"])
        history["train_commitment"].append(train_metrics["commitment"])
        history["train_routing_acc"].append(train_metrics["routing_acc"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_routing_acc"].append(val_metrics["routing_acc"])
        history["cls_weight"].append(cls_weight)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train: {train_metrics['total']:.4f} (recon={train_metrics['recon']:.4f}, "
              f"cls={train_metrics['cls']:.4f}, cls_w={cls_weight:.3f}) | "
              f"Val recon: {val_metrics['recon']:.4f} | "
              f"Route: train={train_metrics['routing_acc']:.3f} val={val_metrics['routing_acc']:.3f} | "
              f"LR: {current_lr:.2e}")

        # Fine-tuning simulation
        if val_entries is not None and epoch % ft_sim_every == 0:
            ft_results = simulate_finetune_on_val(
                model, val_entries, global_mean, global_std, device)
            history["ft_sim"].append((epoch, ft_results))
            print(f"  [FT Sim @ epoch {epoch}] (one-known-normal protocol)")
            if ft_results:
                # Print base once
                first_key = next(iter(ft_results))
                base_m = ft_results[first_key]["base"]
                print(f"    BASE:                  acc={base_m['acc']:.3f} prec={base_m['precision']:.3f} rec={base_m['recall']:.3f}")
                for (ep, lr_ft), metrics in sorted(ft_results.items()):
                    ft_m = metrics["ft"]
                    d_acc = ft_m["acc"] - base_m["acc"]
                    d_prec = ft_m["precision"] - base_m["precision"]
                    d_rec = ft_m["recall"] - base_m["recall"]
                    print(f"    FT ep={ep:2d} lr={lr_ft:.0e}: acc={ft_m['acc']:.3f}({d_acc:+.3f}) "
                          f"prec={ft_m['precision']:.3f}({d_prec:+.3f}) "
                          f"rec={ft_m['recall']:.3f}({d_rec:+.3f})")

        # Reset best tracker when annealing completes — models before and after
        # annealing are not comparable since the loss landscape changes
        if epoch == cls_anneal_epochs:
            best_val_recon = float("inf")
            epochs_without_improvement = 0
            print(f"  [Cls annealing complete — reset early stopping tracker]")

        # Early stopping on val recon loss, only after annealing
        if epoch >= cls_anneal_epochs:
            if val_metrics["recon"] < best_val_recon:
                best_val_recon = val_metrics["recon"]
                epochs_without_improvement = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_recon": best_val_recon,
                    "val_routing_acc": val_metrics["routing_acc"],
                    "history": history,
                }, best_path)
                print(f"  -> Saved best model (val_recon={best_val_recon:.6f}, "
                      f"route_acc={val_metrics['routing_acc']:.3f})")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"(val_recon={checkpoint['val_recon']:.6f}, "
          f"route_acc={checkpoint['val_routing_acc']:.3f})")
    return model, history
