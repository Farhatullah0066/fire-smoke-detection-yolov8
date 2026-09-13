"""
src/data/visualize_samples.py
Draws YOLO-format bounding boxes on random sample images for a sanity check.
Saves annotated samples to results/plots/sample_annotations.png
"""

import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw/train")
OUT_PATH = Path("results/plots/sample_annotations.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "smoke", 1: "fire"}
COLORS = {0: (128, 128, 128), 1: (0, 0, 255)}  # BGR: smoke=gray, fire=red

N_SAMPLES = 9


def draw_boxes(img_path: Path, lbl_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, bw, bh = map(float, parts)
                cls_id = int(cls_id)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                color = COLORS.get(cls_id, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, CLASS_NAMES.get(cls_id, str(cls_id)),
                            (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    img_dir = RAW_DIR / "images"
    lbl_dir = RAW_DIR / "labels"
    all_images = list(img_dir.glob("*.*"))
    samples = random.sample(all_images, min(N_SAMPLES, len(all_images)))

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, img_path in zip(axes.flatten(), samples):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        annotated = draw_boxes(img_path, lbl_path)
        ax.imshow(annotated)
        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"[INFO] Saved sample visualization to {OUT_PATH}")


if __name__ == "__main__":
    main()