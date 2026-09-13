"""
src/data/download_dataset.py
Downloads the Fire & Smoke (D-Fire) dataset for YOLOv8 training.
"""

import os
import subprocess
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SLUG = "sayedgamal99/smoke-fire-detection-yolo"


def download_from_kaggle(slug: str, dest: Path):
    print(f"[INFO] Downloading dataset '{slug}' from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[ERROR]", result.stderr)
        return False
    return True


def unzip_all(folder: Path):
    for zip_path in folder.glob("*.zip"):
        print(f"[INFO] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(folder)
        os.remove(zip_path)
    print("[INFO] Extraction complete.")


if __name__ == "__main__":
    success = download_from_kaggle(DATASET_SLUG, RAW_DIR)
    if success:
        unzip_all(RAW_DIR)