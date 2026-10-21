"""Slot occupancy classifier.
Primary path (scripts/train_mobilenet.py): MobileNetV2 transfer learning.
Robust CPU vision pipeline: multi-feature perceptron + texture & edge analyzer
which trains rapidly on standard CPU and classifies Occupied vs Empty across both
real aerial photography and synthetic benchmarks.
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
INPUT_SIZE = 32
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "parking_mlp.joblib"
WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "parking_mlp_weights.npz"


def image_to_features(image: np.ndarray) -> np.ndarray:
    """Extract invariant multi-spectral, texture, and edge features for occupancy detection."""
    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    h, w = rgb.shape[:2]

    # 1. 32x32 color downsample (3,072 features)
    tiny = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    f_tiny = tiny.reshape(-1)

    # 2. Color and luminance statistics
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    mean_r, std_r = float(np.mean(rgb[:, :, 0])), float(np.std(rgb[:, :, 0]))
    mean_g, std_g = float(np.mean(rgb[:, :, 1])), float(np.std(rgb[:, :, 1]))
    mean_b, std_b = float(np.mean(rgb[:, :, 2])), float(np.std(rgb[:, :, 2]))
    mean_s, std_s = float(np.mean(hsv[:, :, 1])), float(np.std(hsv[:, :, 1]))

    # 3. Texture and edge density (Laplacian variance and Canny edge ratio)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 35, 110)
    edge_ratio = float(np.mean(edges > 0))

    # 4. Sub-quadrant variances (to capture localized vehicle presence in larger bays)
    sub_h, sub_w = max(2, h // 2), max(2, w // 2)
    quad_stds = []
    for sy in range(2):
        for sx in range(2):
            sub = gray[sy * sub_h : (sy + 1) * sub_h, sx * sub_w : (sx + 1) * sub_w]
            quad_stds.append(float(np.std(sub)) if sub.size > 0 else 0.0)

    stats = np.array([
        mean_r, std_r, mean_g, std_g, mean_b, std_b,
        mean_s, std_s, lap_var, edge_ratio, *quad_stds
    ], dtype=np.float32)

    return np.concatenate([f_tiny, stats])


def augment_features(image: np.ndarray) -> list[np.ndarray]:
    """Apply photometric and geometric augmentations to bolster real-world generalization."""
    variants = [image, cv2.flip(image, 1)]
    bright = np.clip(image.astype(np.float32) * 1.20, 0, 255).astype(np.uint8)
    dark = np.clip(image.astype(np.float32) * 0.80, 0, 255).astype(np.uint8)
    variants.extend([bright, dark])

    # Slight rotation jitter
    angle = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 6, 1.0)
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
                    max_iter=100,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )


def export_pure_numpy_weights(pipeline: Pipeline, path: Path = WEIGHTS_PATH) -> None:
    """Export trained scaler and MLP layer weights to pure NumPy .npz (zero-pickle dependencies)."""
    scaler = pipeline.named_steps["scaler"]
    mlp = pipeline.named_steps["mlp"]

    save_dict = {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "num_layers": np.array([len(mlp.coefs_)], dtype=np.int32),
        "classes": np.array(CLASS_NAMES),
    }
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        save_dict[f"w_{i}"] = w
        save_dict[f"b_{i}"] = b

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **save_dict)


def save_sklearn_model(model, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASS_NAMES}, path)
    # Also automatically save pure numpy weights
    export_pure_numpy_weights(model, path.parent / "parking_mlp_weights.npz")


def load_sklearn_model(path: Path = MODEL_PATH):
    payload = joblib.load(path)
    return payload["model"], tuple(payload["classes"])
