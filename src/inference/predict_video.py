"""
src/inference/predict_video.py
Runs the trained YOLOv8 fire/smoke model on a video file
and saves an annotated output video.
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run fire/smoke detection on a video")
    parser.add_argument("--weights", type=str, default="models/best.pt",
                         help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                         help="Path to input video file")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Confidence threshold for detections")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference image size")
    parser.add_argument("--out", type=str, default="results/predictions",
                         help="Output directory for annotated video")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path.resolve()}")
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found at {source_path.resolve()}")

    print(f"[INFO] Loading model from {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] Running inference on video: {source_path}")
    print(f"[INFO] This may take a while on CPU depending on video length...")

    start_time = time.time()

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        name="video_run",
        exist_ok=True,
        stream=True,  # process frame-by-frame, memory efficient for video
    )

    frame_count = 0
    detection_count = 0
    for r in results:
        frame_count += 1
        if r.boxes is not None:
            detection_count += len(r.boxes)
        if frame_count % 50 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    elapsed = time.time() - start_time
    print(f"\n[INFO] Done. Processed {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.2f} FPS)")
    print(f"[INFO] Total detections across video: {detection_count}")
    print(f"[INFO] Annotated video saved under runs/detect/video_run/")


if __name__ == "__main__":
    main()