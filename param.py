#Necessary parameters


import os

DSNAME = "coco"

BATCH = 1

USE_TPU = False
MULTI_CORE = False

import torch

DATA_DIR = '../dataset/'
OUT_DIR = '../result/'
MODEL_DIR = '../models/'
CHECKPOINT_DIR = '../checkpoint/'

TRAIN_DIR = DATA_DIR+"train/"  # UPDATE
TEST_DIR = DATA_DIR+"test/" # UPDATE

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# DATA INFORMATION
IMAGE_SIZE = 224
BATCH_SIZE = 1
GRADIENT_PENALTY_WEIGHT = 10
NUM_EPOCHS = 10
KEEP_CKPT = 2
# save_model_path = MODEL_DIR


if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("running on the CPU")
