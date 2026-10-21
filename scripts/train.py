"""Train the occupancy classifier (sklearn MLP fallback that runs on any CPU).
For MobileNetV2 transfer learning (assignment-preferred), use:
    python scripts/train_mobilenet.py
on a machine where TensorFlow imports successfully (Colab / Streamlit Cloud).
"""
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
from src.model import CLASS_NAMES, augment_features, build_sklearn_model, image_to_features, save_sklearn_model
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
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--epochs", type=int, default=80, help="MLP max_iter (maps to assignment epoch budget)")
    args = parser.parse_args()
    x_train, y_train = load_split(args.splits_dir / "train", augment=True)
    x_val, y_val = load_split(args.splits_dir / "val", augment=False)
    x_test, y_test = load_split(args.splits_dir / "test", augment=False)
    model = build_sklearn_model()
    model.named_steps["mlp"].max_iter = args.epochs
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    metrics = {
        "approach": "crop-based classification (sklearn MLP fallback; see train_mobilenet.py)",
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "classes": list(CLASS_NAMES),
        "input_size": 64,
        "train_samples_with_augmentation": int(len(y_train)),
    }
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, test_pred, target_names=CLASS_NAMES))
    matrix = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Parking slot confusion matrix")
    plt.tight_layout()
    plt.savefig(models_dir / "confusion_matrix.png", dpi=140)
    save_sklearn_model(model)
    print(f"Saved {models_dir / 'parking_mlp.joblib'}")
if __name__ == "__main__":
    main()
