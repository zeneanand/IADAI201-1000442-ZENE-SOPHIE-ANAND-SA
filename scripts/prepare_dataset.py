"""Resize slot images to 224x224 and split 70/15/15 into train/val/test folders."""
from __future__ import annotations
import argparse
import random
import shutil
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("empty", "occupied")
def collect_images(raw_dir: Path, class_name: str) -> list[Path]:
    folder = raw_dir / class_name
    files = [
        p
        for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    files.sort()
    return files
def split_files(files: list[Path], seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
def save_resized(src: Path, dst: Path, size: int = 224) -> None:
    image = cv2.imread(str(src))
    if image is None:
        raise ValueError(f"Could not read {src}")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=100)
    args = parser.parse_args()
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for class_name in CLASSES:
        files = collect_images(args.raw_dir, class_name)
        if len(files) < args.min_per_class:
            raise SystemExit(
                f"{class_name} has {len(files)} images; need at least {args.min_per_class}. "
                "Place PKLot slot crops in data/raw/empty and data/raw/occupied, "
                "or run: python scripts/generate_synthetic_data.py"
            )
        parts = split_files(files, args.seed)
        for split_name, split_files_ in parts.items():
            for src in split_files_:
                dst = args.out_dir / split_name / class_name / src.name
                save_resized(src, dst)
            print(f"{split_name}/{class_name}: {len(split_files_)} images")
if __name__ == "__main__":
    main()
