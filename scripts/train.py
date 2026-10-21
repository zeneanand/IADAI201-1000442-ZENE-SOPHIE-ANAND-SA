from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import (
    CLASS_NAMES,
    augment_features,
    build_sklearn_model,
    export_pure_numpy_weights,
    image_to_features,
    save_sklearn_model,
)


def load_split(split_dir: Path, augment: bool) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label_idx, name in enumerate(CLASS_NAMES):
        folder = split_dir / name
        for path in sorted(folder.glob("*.jpg")):
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            feats = augment_features(image) if augment else [image_to_features(image)]
            xs.extend(feats)
            ys.extend([label_idx] * len(feats))
    return np.stack(xs), np.array(ys)


def extract_real_aerial_samples(lot_path: Path) -> tuple[list[np.ndarray], list[int]]:
    """Extract augmented real-world aerial parking bay crops for high generalization."""
    if not lot_path.exists():
        return [], []

    lot = cv2.imread(str(lot_path))
    lot_rgb = cv2.cvtColor(lot, cv2.COLOR_BGR2RGB)
    h, w, _ = lot.shape
    rows, cols = 6, 10
    slot_h, slot_w = h / rows, w / cols
    empty_bays = {(1, 2), (4, 3), (4, 6), (4, 8), (4, 9), (5, 8)}

    xs, ys = [], []
    # Bay crops
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * slot_w), int(r * slot_h)
            x2, y2 = int((c + 1) * slot_w), int((r + 1) * slot_h)
            crop = lot_rgb[y1:y2, x1:x2]
            label_idx = 0 if (r, c) in empty_bays else 1
            feats = augment_features(crop)
            xs.extend(feats)
            ys.extend([label_idx] * len(feats))

    # Empty asphalt lane crops (label 0 = empty)
    for ly in [int(1.9 * slot_h), int(3.9 * slot_h)]:
        for lx in range(0, w - int(slot_w), int(slot_w // 2)):
            crop = lot_rgb[ly : ly + int(slot_h * 0.8), lx : lx + int(slot_w * 0.8)]
            if crop.size > 0:
                feats = augment_features(crop)
                xs.extend(feats)
                ys.extend([0] * len(feats))

    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--epochs", type=int, default=100, help="MLP max_iter (epoch budget)")
    args = parser.parse_args()

    x_train, y_train = load_split(args.splits_dir / "train", augment=True)
    x_val, y_val = load_split(args.splits_dir / "val", augment=False)
    x_test, y_test = load_split(args.splits_dir / "test", augment=False)

    # Ingest real aerial samples if present
    aerial_path = ROOT / "data" / "samples" / "aerial_monitored_lot.jpg"
    real_x, real_y = extract_real_aerial_samples(aerial_path)
    if real_x:
        x_train = np.concatenate([x_train, np.stack(real_x)], axis=0)
        y_train = np.concatenate([y_train, np.array(real_y)], axis=0)

    print(f"Training dataset size: {len(x_train)} samples ({np.sum(y_train == 1)} occupied, {np.sum(y_train == 0)} empty)")

    model = build_sklearn_model()
    model.named_steps["mlp"].max_iter = args.epochs
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    val_acc = float(accuracy_score(y_val, val_pred))
    test_acc = float(accuracy_score(y_test, test_pred))
    report = classification_report(y_test, test_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics = {
        "approach": "crop-based classification (enhanced multi-feature MLP on 32x32 color + texture/edge stats)",
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "classes": list(CLASS_NAMES),
        "input_features": int(x_train.shape[1]),
        "train_samples_with_augmentation": int(len(y_train)),
    }

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (models_dir / "classification_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, test_pred, target_names=CLASS_NAMES))

    matrix = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(5.2, 4.4), facecolor="#0b1324")
    ax = plt.gca()
    ax.set_facecolor("#0b1324")
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Sample Count"},
    )
    plt.xlabel("Predicted Label", color="#94a3b8")
    plt.ylabel("Actual Label", color="#94a3b8")
    plt.title("ParkVision AI Confusion Matrix (Test Split)", color="#38bdf8", fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    plt.savefig(models_dir / "confusion_matrix.png", dpi=140, facecolor="#0b1324")
    plt.close()

    save_sklearn_model(model, models_dir / "parking_mlp.joblib")
    export_pure_numpy_weights(model, models_dir / "parking_mlp_weights.npz")
    print(f"Saved {models_dir / 'parking_mlp.joblib'} and {models_dir / 'parking_mlp_weights.npz'}")


if __name__ == "__main__":
    main()
