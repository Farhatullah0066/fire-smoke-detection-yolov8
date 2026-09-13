"""
src/data/verify_annotations.py
Sanity-checks that every image has a matching label file,
every label file is valid YOLO format, and reports class distribution.
"""

from pathlib import Path
from collections import Counter

RAW_DIR = Path("data/raw")
CLASS_NAMES = {0: "smoke", 1: "fire"}


def verify_split(split_dir: Path):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[SKIP] {split_dir} missing images/ or labels/ folder")
        return

    image_files = {p.stem for p in img_dir.glob("*.*")}
    label_files = {p.stem for p in lbl_dir.glob("*.txt")}

    missing_labels = image_files - label_files
    orphan_labels = label_files - image_files

    class_counter = Counter()
    empty_label_files = 0
    malformed = 0

    for lbl_path in lbl_dir.glob("*.txt"):
        with open(lbl_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
        if len(lines) == 0:
            empty_label_files += 1
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                malformed += 1
                continue
            cls_id, x, y, w, h = parts
            cls_id = int(cls_id)
            x, y, w, h = map(float, (x, y, w, h))
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                malformed += 1
                continue
            class_counter[cls_id] += 1

    print(f"\n--- {split_dir.name.upper()} ---")
    print(f"Images: {len(image_files)} | Labels: {len(label_files)}")
    print(f"Images missing labels: {len(missing_labels)}")
    print(f"Orphan label files: {len(orphan_labels)}")
    print(f"Empty label files (background/negative images): {empty_label_files}")
    print(f"Malformed annotation lines: {malformed}")
    print("Class distribution (bounding boxes):")
    for cls_id, count in sorted(class_counter.items()):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        print(f"  {name}: {count} boxes")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        verify_split(RAW_DIR / split)