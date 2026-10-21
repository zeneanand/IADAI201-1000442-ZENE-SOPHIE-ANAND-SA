from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
DEST = ROOT / "data"

CLASSES = ["empty", "occupied"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (DEST / split / cls).mkdir(
            parents=True, exist_ok=True
        )

for cls in CLASSES:
    folder = SOURCE / cls

    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if len(images) < 100:
        raise ValueError(
            f"{cls} has {len(images)} images. "
            "Add at least 100 labeled slot images per class."
        )

    random.shuffle(images)

    n = len(images)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, paths in splits.items():
        for i, source in enumerate(paths):
            destination = (
                DEST / split / cls /
                f"{i:06d}_{source.name}"
            )
            shutil.copy2(source, destination)

        print(f"{cls} / {split}: {len(paths)} images")

print("Dataset split complete.")
