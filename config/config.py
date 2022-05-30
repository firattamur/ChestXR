"""
Configuration hyperparameters for train.
"""

# ----------------------------------------
# Dataset
# ----------------------------------------

DATASET_ROOT = "./dataset"
DVALID_SIZE  = 500
IMAGE_SIZE   = 320

# ----------------------------------------
# Model
# ----------------------------------------

MODEL    = "efficientnet_b3a"

NBATCH   = 32
NWORKERS = 4
BATCH_LOG_INTERVAL = 1000

NEPOCHS  = 200

NCLASSES            = 15
LEARNING_RATE       = 0.001
LOSS_SMOOTH_FACTOR  = 0.10

CHECKPOINTS_PATH    = "/Users/firattamur/Desktop/ChestXR/checkpoints"
CHECKPOINT_SAVE_INTERVAL = 1

PRETRAINEDS_PATH   = "/Users/firattamur/Desktop/ChestXR/pretraineds"
