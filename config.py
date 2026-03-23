import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "physionet.org", "files", "mimic-iv-ecg", "1.0")
RECORD_LIST_CSV = os.path.join(DATA_ROOT, "record_list.csv")
MEASUREMENTS_CSV = os.path.join(DATA_ROOT, "machine_measurements.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# Signal parameters
SAMPLING_RATE_ORIGINAL = 500
SAMPLING_RATE_TARGET = 250
LEAD_INDEX = 1  # Lead II
NUM_LEADS = 12
SAMPLES_PER_RECORDING = 5000
WINDOW_SECONDS = 1.0
WINDOW_SIZE = 250  # SAMPLING_RATE_TARGET * WINDOW_SECONDS
WINDOWS_PER_RECORDING = 10
ADC_GAIN = 200.0

# Model parameters
INPUT_DIM = 250
HIDDEN_DIM = 256
LATENT_DIM = 32

# Training parameters
BATCH_SIZE = 512
LR_BASE = 1e-3
LR_FINETUNE = 1e-4
EPOCHS_BASE = 50
EPOCHS_FINETUNE = 20
KL_WEIGHT = 1.0
WEIGHT_DECAY = 1e-5
PATIENCE = 7  # early stopping patience

# Per-patient fine-tuning
MIN_RECORDINGS_FOR_FINETUNE = 5

# Data splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
RANDOM_SEED = 42
