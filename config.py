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
LATENT_DIM = 16
FRAME_SIZE = 25       # samples per frame (10 frames per window)
FEATURE_DIM = 256     # feature dim for TDNN layers (max hardware I/O)
ENCODER_BLOCKS = 5    # number of TDNN blocks in encoder
DECODER_BLOCKS = 5    # number of TDNN blocks in decoder

# Training parameters
BATCH_SIZE = 512
LR_BASE = 3e-4
LR_FINETUNE = 5e-5
EPOCHS_BASE = 80
EPOCHS_FINETUNE = 20
KL_WEIGHT = 0.2
WEIGHT_DECAY = 1e-5
PATIENCE = 10  # early stopping patience
KL_ANNEAL_EPOCHS = 10  # linearly ramp KL weight over this many epochs

# VQ-VAE parameters
NUM_CODES = 8          # total codebook entries
NORMAL_CODE_IDS = [0]  # which code indices are "normal"
EMBED_DIM = 128        # codebook embedding dimension
COMMITMENT_COST = 1.0  # weight for commitment loss
CODEBOOK_COST = 1.0    # weight for codebook loss
CLS_WEIGHT = 0.5       # max classification loss weight
CLS_ANNEAL_EPOCHS = 20 # ramp cls_weight from 0 to CLS_WEIGHT over this many epochs

# Per-patient fine-tuning
MIN_RECORDINGS_FOR_FINETUNE = 5

# Data splits
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
RANDOM_SEED = 42
