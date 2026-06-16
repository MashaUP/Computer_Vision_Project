from pathlib import Path
import torch

ARCH_TYPE = "xxs"

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
# Training
# ==========================================
BATCH_SIZE = 8
FT_BATCH_SIZE = 8
LEARNING_RATE = 1e-4

EPOCHS_PRETRAIN = 50
EPOCHS_FINETUNE = 60

# ==========================================
# Image resolution
# ==========================================
RGB_img_res = (3, 256, 256)
IMG_SIZE = RGB_img_res[1:]
NYU_SIZE = (192, 256)  # (height, width)
KITTI_SIZE = (192, 636)  # (height, width)

# ==========================================
# Paths
# ==========================================
if torch.cuda.is_available() and torch.cuda.get_device_name(0) == "Tesla T4":
    print("Running on Colab with Tesla T4 GPU. Setting paths for Colab environment.")
    BASE_DIR = Path("/content/drive/MyDrive/CV-project-25-26")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_NYU_PATH = BASE_DIR / "data" / "raw" / "nyu" 

DATASET_KITTI_PATH = BASE_DIR / "data" / "raw" / "kitti" 
# Optional external KITTI GT path
DATASET_KITTI_PATH_GT = BASE_DIR / "data" / "raw" / "kitti" 

# ==========================================
# Output paths
# ==========================================
MODELS_DIR = BASE_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
JSON_DIR = OUTPUT_DIR / "json"

# ==========================================
# Checkpoints
# ==========================================

SSL_CKPT_PATH_KITTI = MODEL_SAVE_PATH.joinpath("pretrain", "last_kitti.ckpt")
SSL_CKPT_PATH_NYU = MODEL_SAVE_PATH.joinpath("pretrain", "last_nyu.ckpt")
FINETUNE_CKPT_PATH = MODEL_SAVE_PATH.joinpath("finetune", "last.ckpt")


# ==========================================
# Required folders
# ==========================================

REQUIRED_FOLDERS = [
    MODEL_SAVE_PATH,
    OUTPUT_DIR, 
    IMAGES_DIR,
    JSON_DIR
]

for path in REQUIRED_FOLDERS:
    path.mkdir(parents=True, exist_ok=True)


# ==========================================
# Debug
# ==========================================
if __name__ == "__main__":
    print("=== Configuration ===")
    print(f"Device: {DEVICE}")
    print(f"Image size: {RGB_img_res}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Checkpoint path: {MODEL_SAVE_PATH}")