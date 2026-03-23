# Project Structure — ECG Anomaly Detection with Per-Patient VAE Fine-Tuning

## Overview

A Variational Autoencoder (VAE) trained on MIMIC-IV-ECG data to detect anomalous cardiac rhythms. The core hypothesis is that per-patient fine-tuning in an unsupervised fashion reduces false alarms and increases recall compared to a population-level base model.

## Directory Layout

```
Hyperdrive/
├── config.py                              # Central configuration
├── requirements.txt                       # Python dependencies
├── DATA_ANALYSIS.md                       # Dataset structure analysis
├── PROJECT_STRUCTURE.md                   # This file
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py               # Signal loading and windowing
│   │   ├── labeling.py                     # Label extraction from reports
│   │   ├── dataset.py                      # PyTorch Dataset classes
│   │   └── splits.py                       # Patient-level data splitting
│   │
│   ├── models/
│   │   ├── vae.py                          # VAE architecture
│   │   └── loss.py                         # MSE + KL divergence loss
│   │
│   ├── training/
│   │   ├── train_base.py                   # Base model training loop
│   │   └── finetune_patient.py             # Per-patient fine-tuning
│   │
│   └── evaluation/
│       ├── anomaly_scoring.py              # Reconstruction + latent scoring
│       └── metrics.py                      # AUROC, AUPRC, F1, comparison
│
├── scripts/
│   ├── 01_build_cache.py                   # Pre-compute labels and stats
│   ├── 02_train_base.py                    # Train the base VAE
│   └── 03_finetune_and_evaluate.py         # Fine-tune + compare results
│
├── cache/                                  # Generated: cached labels and stats
├── checkpoints/                            # Generated: model weights
└── physionet.org/                          # Data (git-ignored)
```

## Model Architecture

```
Input (250)  ──→  FC+Tanh (256)  ──→  FC+Tanh (256)  ──┬──→  μ (32)
                                                        └──→  log σ² (32)
                                                                │
                                                          Reparameterize
                                                                │
                                                             z (32)
                                                                │
Output (250) ←──  FC (256)  ←──  FC+Tanh (256)  ←──  FC+Tanh (256)
```

- **Input**: 250 samples (1 second of Lead II at 250 Hz)
- **Hidden layers**: 256×256 fully-connected with tanh activations
- **Latent dimension**: 32
- **Total parameters**: 284,986
- **Loss**: MSE reconstruction + KL divergence

## Data Pipeline

```
.dat file (120 KB, 12 leads, 500 Hz, 10s)
    │
    ├── Extract Lead II (index 1)
    ├── Convert to mV (÷ 200.0)
    ├── Downsample 500 Hz → 250 Hz
    └── Slice into 10 × 250-sample windows (1 second each)
```

## Labeling

Binary labels derived from machine-generated reports in `machine_measurements.csv`:

| Label    | Criteria                                                        |
| -------- | --------------------------------------------------------------- |
| Normal   | Any report field starts with "Normal ECG" or "Sinus rhythm"     |
| Abnormal | Everything else (e.g., atrial fibrillation, ST elevation, etc.) |

Labels are per-recording (10-second) and inherited by all 10 windows within.

## Training Pipeline

### Step 1: Build Cache (`01_build_cache.py`)

- Parses 800K records from CSVs
- Assigns normal/abnormal labels
- Computes global signal mean and std from 10K sampled recordings
- Saves to `cache/dataset_cache.pkl`

### Step 2: Train Base Model (`02_train_base.py`)

- Patient-level 70/10/20 train/val/test split (no patient leakage)
- Trains VAE on all training recordings
- Adam optimizer (lr=1e-3, weight decay=1e-5)
- ReduceLROnPlateau scheduler
- Early stopping (patience=7)
- Saves best checkpoint to `checkpoints/base_model.pt`

### Step 3: Fine-Tune and Evaluate (`03_finetune_and_evaluate.py`)

- Selects test patients with ≥5 recordings and both normal + abnormal labels
- For each patient:
  1. Temporally splits recordings (early 60% normals → fine-tune, rest → evaluate)
  2. Fine-tunes a copy of the base model (lr=1e-4, 20 epochs, encoder fc1 frozen)
  3. Scores held-out recordings with both base and fine-tuned models
- Reports per-patient and aggregate metrics

## Anomaly Detection

Two scoring methods per recording:

1. **Reconstruction error**: MSE between input and output, max across 10 windows
2. **Latent distance**: L2 norm of the latent mean encoding, max across 10 windows

## Key Hyperparameters

| Parameter          | Value  | Location   |
| ------------------ | ------ | ---------- |
| `INPUT_DIM`        | 250    | config.py  |
| `HIDDEN_DIM`       | 256    | config.py  |
| `LATENT_DIM`       | 32     | config.py  |
| `BATCH_SIZE`       | 512    | config.py  |
| `LR_BASE`          | 1e-3   | config.py  |
| `LR_FINETUNE`      | 1e-4   | config.py  |
| `KL_WEIGHT`        | 1.0    | config.py  |
| `EPOCHS_BASE`      | 50     | config.py  |
| `EPOCHS_FINETUNE`  | 20     | config.py  |
| `PATIENCE`         | 7      | config.py  |

## Run Order

```bash
pip install -r requirements.txt
python scripts/01_build_cache.py
python scripts/02_train_base.py
python scripts/03_finetune_and_evaluate.py
```
