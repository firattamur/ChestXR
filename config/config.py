"""
Configuration hyperparameters for train.
"""

# ----------------------------------------
# Dataset
# ----------------------------------------

DATASET_ROOT = "./dataset"
DVALID_SIZE  = 1000

# ----------------------------------------
# Model
# ----------------------------------------

MODEL    = "mobilenetv3_large_100"

NBATCH   = 64
NWORKERS = 4
BATCH_LOG_INTERVAL = 1000

NEPOCHS  = 200

NCLASSES            = 14
LEARNING_RATE       = 0.001
LOSS_SMOOTH_FACTOR  = 0.10

CHECKPOINTS_PATH    = "./checkpoints"
CHECKPOINT_SAVE_INTERVAL = 1

PRETRAINEDS_PATH   = "./pretraineds"