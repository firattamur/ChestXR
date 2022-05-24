"""
Configuration hyperparameters for train.
"""

# ----------------------------------------
# Dataset
# ----------------------------------------

DATASET_ROOT       = "./dataset"
DATASET_PATH_TEST  = "./dataset/images/train/images-test"
DATASET_PATH_VALID = "./dataset/images/train/images-valid"
DATASET_PATH_TRAIN = "./dataset/images/train/images-train"

# ----------------------------------------
# Model
# ----------------------------------------

MODEL    = "efficientnet-lite2"

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