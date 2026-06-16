import torch
from pathlib import Path

# ==========================================
# Device
# ==========================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ==========================================
# Architecture
# ==========================================
# Variants: "xxs" (0.71M), "xs" (1.45M), "s" (3.29M)  — paper Table II
ARCH_TYPE = "xxs"

# ==========================================
# MobileViT encoder dimensions
# ==========================================
ENC_DIMS = {
    "xxs": 160,
    "xs": 192,
    "s": 320,
}

# ==========================================
# Image resolutions  (paper Section IV-B)
# ==========================================
# SSL pre-training: square crop used by LeJEPA multi-crop
RGB_img_res = (3, 256, 256)   # (C, H, W)  — used by network.py
IMG_SIZE    = RGB_img_res[1:]  # (H, W)

# Fine-tuning resolutions per dataset
NYU_SIZE   = (192, 256)   # (H, W) — 640×480 downsampled
KITTI_SIZE = (192, 636)   # (H, W) — Eigen split

# ==========================================
# Training hyper-parameters  (paper Section IV-A)
# ==========================================
# --- Supervised fine-tuning ---
SSL_BATCH_SIZE = 32
FT_BATCH_SIZE  = 64
LEARNING_RATE = 1e-3          # AdamW lr = 0.001 as in paper
WEIGHT_DECAY  = 0.01          # wd = 0.01 as in paper
EPOCHS_FINETUNE = 60          # 60 epochs as in paper
LR_STEP_SIZE  = 20            # StepLR: decrement 0.1 every 20 epochs
LR_GAMMA      = 0.1

# --- LeJEPA SSL pre-training ---
EPOCHS_PRETRAIN = 50          # self-supervised pre-training epochs
SSL_LR          = 5e-4        # LeJEPA learning rate
SSL_WEIGHT_DECAY = 5e-2       # LeJEPA weight decay
SSL_PROJ_DIM    = 64          # projector output dimension
SSL_NUM_SLICES  = 256         # SIGReg slices
SSL_LAMB        = 0.05       # SIGReg regularisation weight

# ==========================================
# Balanced Loss Function  (paper Section III-B, Eq. 1)
# ==========================================
# lambda_1=0.5, lambda_2=100, lambda_3=100
# val_range: 1000 (cm) for NYU, 80 (m) for KITTI
LAMBDA_1 = 0.5
LAMBDA_2 = 100
LAMBDA_3 = 100

# ==========================================
# Data augmentation  (paper Section III-C)
# ==========================================
# All random transforms applied with probability 0.5
AUG_PROB = 0.5
# D-shift range: ±10 cm (NYU) / ±10 dm (KITTI)
DEPTH_SHIFT_RANGE = 10

# ==========================================
# Paths
# ==========================================
if torch.cuda.is_available() and torch.cuda.get_device_name(0) == "Tesla T4":
    print("[INFO] Running on Colab — Tesla T4 detected. Using Drive paths.")
    BASE_DIR = Path("/content/drive/MyDrive/CV-project-25-26")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
DATASET_NYU_PATH   = BASE_DIR / "data" / "raw" / "nyu"
DATASET_KITTI_PATH = BASE_DIR / "data" / "raw" / "kitti"
DATASET_KITTI_PATH_GT = BASE_DIR / "data" / "raw" / "kitti"   # optional external GT

# ==========================================
# Output paths
# ==========================================
MODELS_DIR        = BASE_DIR / "models"
MODEL_SAVE_PATH   = MODELS_DIR / "checkpoints"
OUTPUT_DIR        = BASE_DIR / "output"
IMAGES_DIR        = OUTPUT_DIR / "images"
JSON_DIR          = OUTPUT_DIR / "json"

# Checkpoint shortcuts
SSL_CKPT_PATH = MODEL_SAVE_PATH / "ssl_nyu" / "last.ckpt"
FINETUNE_CKPT_PATH = MODEL_SAVE_PATH / "finetune" / "last.ckpt"

# ==========================================
# Create required folders at import time
# ==========================================
REQUIRED_FOLDERS = [
    MODEL_SAVE_PATH,
    MODEL_SAVE_PATH / "ssl_nyu",
    MODEL_SAVE_PATH / "ssl_kitti",
    MODEL_SAVE_PATH / "finetune",
    OUTPUT_DIR,
    IMAGES_DIR,
    JSON_DIR,
]

for _path in REQUIRED_FOLDERS:
    _path.mkdir(parents=True, exist_ok=True)


# ==========================================
# Debug
# ==========================================
if __name__ == "__main__":
    print("=== Configuration ===")
    print(f"  Device          : {DEVICE}")
    print(f"  Arch type       : {ARCH_TYPE}")
    print(f"  SSL img size    : {RGB_img_res}")
    print(f"  NYU  fine-tune  : {NYU_SIZE}")
    print(f"  KITTI fine-tune : {KITTI_SIZE}")
    print(f"  Batch size SSL  : {SSL_BATCH_SIZE}")
    print(f"  Batch size FT   : {FT_BATCH_SIZE}")
    print(f"  LR (finetune)   : {LEARNING_RATE}")
    print(f"  Epochs pretrain : {EPOCHS_PRETRAIN}")
    print(f"  Epochs finetune : {EPOCHS_FINETUNE}")
    print(f"  Checkpoint dir  : {MODEL_SAVE_PATH}")
    print(f"  Output dir      : {OUTPUT_DIR}")