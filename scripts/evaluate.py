"""Evaluate the trained model on the held-out test split."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import load_model, predict_slot
from src.model import CLASS_NAMES
def main() -> None:
    model = load_model()
    y_true, y_pred = [], []
    for idx, name in enumerate(CLASS_NAMES):
        for path in sorted((ROOT / "data" / "splits" / "test" / name).glob("*.jpg")):
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            label, _ = predict_slot(model, image)
            y_true.append(idx)
            y_pred.append(CLASS_NAMES.index(label))
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    (ROOT / "models" / "classification_report.json").write_text(json.dumps(report, indent=2))
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Parking slot confusion matrix")
    plt.tight_layout()
    out = ROOT / "models" / "confusion_matrix.png"
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")
if __name__ == "__main__":
    main()
