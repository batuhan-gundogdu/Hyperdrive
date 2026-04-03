import os
import copy
import torch
from torch.utils.data import DataLoader

import config
from src.models.loss import vae_loss


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

    # Freeze early encoder conv layers to preserve learned feature extraction
    for param in model.encoder.conv.parameters():
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


def finetune_and_save(base_model, patient_dataset, subject_id, device, **kwargs):
    """Fine-tune and save the per-patient model checkpoint."""
    model = finetune_for_patient(base_model, patient_dataset, device, **kwargs)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(config.CHECKPOINT_DIR, f"patient_{subject_id}.pt")
    torch.save(model.state_dict(), path)
    return model
