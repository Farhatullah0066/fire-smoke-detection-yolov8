"""
src/training/train.py
Trains a YOLOv8 model on the Fire & Smoke dataset.
Configured for CPU-only training on modest hardware (i7, no dedicated GPU).
"""

import argparse
import sys
import traceback
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on fire/smoke dataset")
    parser.add_argument("--data", type=str, default="data/raw/data.yaml",
                         help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Base pretrained model (nano = fastest for CPU)")
    parser.add_argument("--epochs", type=int, default=10,
                         help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=320,
                         help="Image size (smaller = faster on CPU)")
    parser.add_argument("--batch", type=int, default=8,
                         help="Batch size (keep small for CPU)")
    parser.add_argument("--fraction", type=float, default=1.0,
                         help="Fraction of training data to use (e.g. 0.1 = 10%%, for fast smoke tests)")
    parser.add_argument("--name", type=str, default="fire_smoke_run",
                         help="Name for this training run (saved under runs/detect/<name>)")
    parser.add_argument("--patience", type=int, default=15,
                         help="Early stopping patience (epochs with no improvement)")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_path.resolve()}")

    print(f"[INFO] Loading base model: {args.model}", flush=True)
    model = YOLO(args.model)

    print(f"[INFO] Starting training:", flush=True)
    print(f"       data     = {data_path}", flush=True)
    print(f"       epochs   = {args.epochs}", flush=True)
    print(f"       imgsz    = {args.imgsz}", flush=True)
    print(f"       batch    = {args.batch}", flush=True)
    print(f"       fraction = {args.fraction}", flush=True)
    print(f"       device   = cpu", flush=True)

    try:
        results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            fraction=args.fraction,
            device="cpu",
            workers=0,
            patience=args.patience,
            name=args.name,
            verbose=True,
            plots=True,
        )
        print(f"[INFO] Training finished successfully.", flush=True)
    except Exception as e:
        print(f"[ERROR] Training raised an exception: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("[INFO] Training complete. Check runs/detect/<name>/ for results.", flush=True)


if __name__ == "__main__":
    main()