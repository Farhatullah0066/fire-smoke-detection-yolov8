"""
src/evaluation/evaluate.py
Runs full evaluation of a trained YOLOv8 model on the test set.
Reports mAP50, mAP50-95, Precision, Recall (overall and per-class),
and saves confusion matrix + PR curves.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to trained model weights (best.pt)")
    parser.add_argument("--data", type=str, default="data/raw/data.yaml",
                         help="Path to dataset YAML config")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Image size used during evaluation")
    parser.add_argument("--split", type=str, default="test",
                         choices=["val", "test"],
                         help="Which split to evaluate on")
    parser.add_argument("--name", type=str, default="eval_test",
                         help="Name for this evaluation run")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path.resolve()}")

    print(f"[INFO] Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] Running evaluation on '{args.split}' split...")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        split=args.split,
        name=args.name,
        plots=True,
        save_json=True,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\nPer-class results:")
    class_names = model.names
    for i, class_id in enumerate(metrics.ap_class_index):
        name = class_names[class_id]
        p, r, ap50, ap = metrics.box.p[i], metrics.box.r[i], metrics.box.ap50[i], metrics.box.ap[i]
        print(f"  {name:10s} | P: {p:.4f} | R: {r:.4f} | mAP50: {ap50:.4f} | mAP50-95: {ap:.4f}")

    print(f"\n[INFO] Plots and results saved under runs/detect/{args.name}/")


if __name__ == "__main__":
    main()