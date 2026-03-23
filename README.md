# Hyperdrive: Per-Patient ECG Anomaly Detection with Variational Autoencoders

Unsupervised anomaly detection on 12-lead ECG recordings from the [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/) dataset. A Variational Autoencoder learns to reconstruct normal cardiac rhythms, then flags recordings with high reconstruction error as anomalous. The key contribution is showing that **per-patient fine-tuning reduces false alarms and increases recall** compared to a population-level model.

## Motivation

Population-level anomaly detectors suffer from high false alarm rates because patients have naturally different cardiac morphologies. What looks abnormal for the average patient may be perfectly normal for an individual (e.g., athlete's heart, benign early repolarization). By fine-tuning the VAE on each patient's own normal recordings, the model learns their personal baseline and only flags true deviations.

## Dataset

MIMIC-IV-ECG v1.0: 800,035 diagnostic ECG recordings from 161,352 patients.

| Property         | Value                              |
| ---------------- | ---------------------------------- |
| Leads            | 12 (I, II, III, aVR, aVF, aVL, V1-V6) |
| Sampling rate    | 500 Hz                             |
| Duration         | 10 seconds per recording           |
| Patients         | 161,352                            |
| Total recordings | 800,035                            |
| Studies/patient  | median 2, mean ~5, max 260         |

See [DATA_ANALYSIS.md](DATA_ANALYSIS.md) for the full data structure breakdown.

## Approach

### Preprocessing

Each 10-second recording is processed into 10 one-second windows:

1. Extract **Lead II** (best single lead for rhythm analysis)
2. Downsample 500 Hz to **250 Hz**
3. Slice into **10 windows of 250 samples** each
4. Z-score normalize using global statistics

### Model

A fully-connected Variational Autoencoder with tanh activations:

```
Input (250) → FC+Tanh (256) → FC+Tanh (256) → μ (32), log σ² (32)
                                                        ↓
                                                  Reparameterize → z (32)
                                                        ↓
Output (250) ← FC (256) ← FC+Tanh (256) ← FC+Tanh (256)
```

- **Parameters**: 284,986
- **Loss**: MSE reconstruction + KL divergence
- **Anomaly score**: reconstruction error (max across windows per recording)

### Training Strategy

1. **Base model**: Train on the full training set (all patients, all recordings)
2. **Per-patient fine-tuning**: For each patient, fine-tune a copy of the base model on their normal recordings only (encoder first layer frozen, lr=1e-4, 20 epochs)
3. **Evaluation**: Compare base vs fine-tuned anomaly detection on held-out recordings

Patient-level data splitting ensures no patient appears in both train and test sets.

## Project Structure

```
Hyperdrive/
├── config.py                     # Hyperparameters and paths
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Signal loading, downsampling, windowing
│   │   ├── labeling.py           # Normal/abnormal from machine reports
│   │   ├── dataset.py            # PyTorch Dataset classes
│   │   └── splits.py             # Patient-level train/val/test splits
│   ├── models/
│   │   ├── vae.py                # Encoder, Decoder, VAE
│   │   └── loss.py               # MSE + KL loss
│   ├── training/
│   │   ├── train_base.py         # Base model training loop
│   │   └── finetune_patient.py   # Per-patient fine-tuning
│   └── evaluation/
│       ├── anomaly_scoring.py    # Reconstruction + latent scoring
│       └── metrics.py            # AUROC, AUPRC, F1, base vs fine-tuned
├── scripts/
│   ├── 01_build_cache.py         # Pre-compute labels and signal stats
│   ├── 02_train_base.py          # Train the base VAE
│   └── 03_finetune_and_evaluate.py  # Fine-tune + compare results
└── physionet.org/                # Data directory (git-ignored)
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation of each component.

## Quick Start

### 1. Data

Download MIMIC-IV-ECG v1.0 from [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) into the `physionet.org/` directory (requires credentialed access).

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
# Build label cache and compute signal statistics
python scripts/01_build_cache.py

# Train base VAE on full training set
python scripts/02_train_base.py

# Fine-tune per patient and evaluate
python scripts/03_finetune_and_evaluate.py
```

## Labeling

Binary labels are derived from machine-generated cardiology reports in `machine_measurements.csv`:

- **Normal**: any report field starts with "Normal ECG" or "Sinus rhythm"
- **Abnormal**: everything else (atrial fibrillation, ST elevation, conduction abnormalities, etc.)

Labels apply to the full 10-second recording and are inherited by all 10 one-second windows.

## Evaluation

The fine-tuning evaluation script (`03_finetune_and_evaluate.py`) selects test patients with at least 5 recordings and both normal and abnormal labels. For each patient:

1. Early normal recordings are used for fine-tuning (temporal split)
2. Remaining recordings are scored by both the base and fine-tuned models
3. Per-patient and aggregate metrics are reported

Metrics: AUROC, AUPRC, precision, recall, F1.

## Key Hyperparameters

| Parameter        | Value | Description                      |
| ---------------- | ----- | -------------------------------- |
| `LATENT_DIM`     | 32    | VAE latent space dimensionality  |
| `HIDDEN_DIM`     | 256   | Fully-connected layer width      |
| `LR_BASE`        | 1e-3  | Base model learning rate         |
| `LR_FINETUNE`    | 1e-4  | Fine-tuning learning rate        |
| `KL_WEIGHT`      | 1.0   | Beta-VAE KL divergence weight    |
| `EPOCHS_BASE`    | 50    | Max base training epochs         |
| `EPOCHS_FINETUNE`| 20    | Fine-tuning epochs per patient   |
| `PATIENCE`       | 7     | Early stopping patience          |

All configurable in [config.py](config.py).
