"""
src/inference/predict_image.py
Runs the trained YOLOv8 fire/smoke model on a single image
and saves an annotated output image.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run fire/smoke detection on an image")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                         help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                         help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Confidence threshold for detections")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference image size")
    parser.add_argument("--out", type=str, default="results/predictions",
                         help="Output directory for annotated image")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path.resolve()}")
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found at {source_path.resolve()}")

    print(f"[INFO] Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] Running inference on {source_path}")
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        imgsz=args.imgsz,
        save=False,  # we'll save manually to control the output path
    )

    out_dir = Path(args.out) / "image_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print("[INFO] No fire or smoke detected.")
        else:
            print(f"[INFO] Detected {len(boxes)} object(s):")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                print(f"  - {label}: confidence {conf:.2f}")

        out_path = out_dir / source_path.name
        r.save(filename=str(out_path))

    print(f"\n[INFO] Annotated image saved to {out_dir / source_path.name}")


if __name__ == "__main__":
    main()