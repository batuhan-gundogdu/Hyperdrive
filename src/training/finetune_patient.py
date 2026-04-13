import os
import copy
import torch
from torch.utils.data import DataLoader

import config
from src.models.loss import vae_loss, vqvae_loss


def finetune_for_patient(base_model, patient_dataset, device,
                         epochs=config.EPOCHS_FINETUNE, lr=config.LR_FINETUNE,
                         kl_weight=config.KL_WEIGHT, batch_size=64):
    """
    Fine-tune a copy of the base model on a single patient's normal recordings.

    Freezes the first encoder layer (fc1) to preserve learned feature extraction.
    Fine-tunes fc2, fc_mu, fc_logvar, and the full decoder.
    """
    if len(patient_dataset) == 0:
        return base_model

    model = copy.deepcopy(base_model)
    model.to(device)

    # Freeze early encoder TDNN blocks to preserve learned feature extraction
    # Keep frame_embed and first half of TDNN blocks frozen
    for param in model.encoder.frame_embed.parameters():
        param.requires_grad = False
    for param in model.encoder.embed_norm.parameters():
        param.requires_grad = False
    n_freeze = len(model.encoder.blocks) // 2
    for block in model.encoder.blocks[:n_freeze]:
        for param in block.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=config.WEIGHT_DECAY)

    loader = DataLoader(patient_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            x = batch["signal"].to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(x_recon, x, mu, logvar, kl_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

    return model


def finetune_vqvae_for_patient(base_model, patient_dataset, device,
                               epochs=config.EPOCHS_FINETUNE, lr=config.LR_FINETUNE,
                               batch_size=64, freeze_encoder=False,
                               commitment_cost=config.COMMITMENT_COST):
    """
    Fine-tune a VQ-VAE for a specific patient.

    Uses ALL of the patient's recordings (label-agnostic).
    Freezes: codebook always.
    freeze_encoder=False: freeze codebook + early encoder blocks (default)
    freeze_encoder=True:  freeze codebook + entire encoder (decoder-only fine-tuning)
    """
    if len(patient_dataset) == 0:
        return base_model

    model = copy.deepcopy(base_model)
    model.to(device)

    # Freeze the codebook — preserve learned normal/abnormal codes
    for param in model.vq.codebook.parameters():
        param.requires_grad = False

    if freeze_encoder:
        # Freeze entire encoder — only decoder adapts
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        # Freeze early encoder blocks only
        for param in model.encoder.frame_embed.parameters():
            param.requires_grad = False
        for param in model.encoder.embed_norm.parameters():
            param.requires_grad = False
        n_freeze = len(model.encoder.blocks) // 2
        for block in model.encoder.blocks[:n_freeze]:
            for param in block.parameters():
                param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=config.WEIGHT_DECAY)

    loader = DataLoader(patient_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        for batch in loader:
            x = batch["signal"].to(device)
            optimizer.zero_grad()
            # No labels — unmasked routing during fine-tuning
            x_recon, z_e, z_q, indices, cb_loss, commit_loss, cls_loss = model(x, labels=None)
            # No codebook loss (codebook is frozen), no cls loss, only recon + commitment
            zero = torch.tensor(0.0, device=device)
            loss, _, _, _, _ = vqvae_loss(x_recon, x, zero, commit_loss, zero,
                                          commitment_cost=commitment_cost)
            loss.backward()
            optimizer.step()

    return model


def finetune_vqvae_one_normal(base_model, normal_dataset, device,
                              epochs=config.EPOCHS_FINETUNE, lr=config.LR_FINETUNE,
                              batch_size=64,
                              commitment_cost=config.COMMITMENT_COST):
    """
    Fine-tune a VQ-VAE on one (or few) known-normal recordings for a patient.

    Uses reconstruction + commitment + classification loss with all labels=0
    (normal), forcing the encoder to route this patient's normal pattern
    to the normal code(s).

    Codebook is frozen. Early encoder blocks are frozen.
    """
    if len(normal_dataset) == 0:
        return base_model

    model = copy.deepcopy(base_model)
    model.to(device)

    for param in model.vq.codebook.parameters():
        param.requires_grad = False
    for param in model.encoder.frame_embed.parameters():
        param.requires_grad = False
    for param in model.encoder.embed_norm.parameters():
        param.requires_grad = False
    n_freeze = len(model.encoder.blocks) // 2
    for block in model.encoder.blocks[:n_freeze]:
        for param in block.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=config.WEIGHT_DECAY)

    loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        for batch in loader:
            x = batch["signal"].to(device)
            # All labels are 0 (normal) — we KNOW these are normal recordings
            labels = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            optimizer.zero_grad()
            x_recon, z_e, z_q, indices, cb_loss, commit_loss, cls_loss = model(x, labels=labels)
            zero = torch.tensor(0.0, device=device)
            loss, _, _, _, _ = vqvae_loss(x_recon, x, zero, commit_loss, cls_loss,
                                          commitment_cost=commitment_cost)
            loss.backward()
            optimizer.step()

    return model


def finetune_and_save(base_model, patient_dataset, subject_id, device, **kwargs):
    """Fine-tune and save the per-patient model checkpoint."""
    model = finetune_for_patient(base_model, patient_dataset, device, **kwargs)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(config.CHECKPOINT_DIR, f"patient_{subject_id}.pt")
    torch.save(model.state_dict(), path)
    return model
