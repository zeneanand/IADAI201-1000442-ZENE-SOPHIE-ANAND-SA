"""Slot occupancy classifier.
Primary path (scripts/train_mobilenet.py): MobileNetV2 transfer learning.
CPU fallback used in this repo: a multi-layer perceptron on 64×64 RGB crops,
which trains without AVX/TensorFlow and still classifies Occupied vs Empty.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
CLASS_NAMES = ("empty", "occupied")
INPUT_SIZE = 64
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "parking_mlp.joblib"
def image_to_features(image: np.ndarray) -> np.ndarray:
    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(-1)
def augment_features(image: np.ndarray) -> list[np.ndarray]:
    variants = [image, cv2.flip(image, 1)]
    bright = np.clip(image.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
    dark = np.clip(image.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    variants.extend([bright, dark])
    angle = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 8, 1.0)
    rotated = cv2.warpAffine(image, angle, (image.shape[1], image.shape[0]))
    variants.append(rotated)
    return [image_to_features(v) for v in variants]
def build_sklearn_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=80,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
def save_sklearn_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASS_NAMES}, path)
def load_sklearn_model(path: Path = MODEL_PATH):
    payload = joblib.load(path)
    return payload["model"], tuple(payload["classes"])
