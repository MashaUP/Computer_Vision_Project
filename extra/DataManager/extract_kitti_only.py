from pathlib import Path
import zipfile

RGB_DIR = Path("./data/raw/kitti/rgb")

def extract_rgb_only():
    for zip_file in RGB_DIR.rglob("*.zip"):

        print(f"[EXTRACT] {zip_file.name}")

        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(RGB_DIR)

extract_rgb_only()