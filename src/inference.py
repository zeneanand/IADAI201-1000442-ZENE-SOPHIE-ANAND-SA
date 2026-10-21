"""Load the trained classifier and predict slot occupancy."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.model import CLASS_NAMES, MODEL_PATH, image_to_features, load_sklearn_model
from src.preprocessing import split_into_slots
ROOT = Path(__file__).resolve().parents[1]
KERAS_PATH = ROOT / "models" / "parking_mobilenet.keras"
@lru_cache(maxsize=1)
def load_model():
    if KERAS_PATH.exists():
        try:
            from tensorflow import keras
            from src.preprocessing import to_model_input
            keras_model = keras.models.load_model(KERAS_PATH)
            class KerasAdapter:
                backend = "mobilenet"
                def predict_slot(self, image: np.ndarray) -> tuple[str, float]:
                    probs = keras_model.predict(to_model_input(image), verbose=0)[0]
                    idx = int(np.argmax(probs))
                    return CLASS_NAMES[idx], float(probs[idx])
            return KerasAdapter()
        except Exception:
            pass
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found. Run: python scripts/train.py\nLooked for {MODEL_PATH} and {KERAS_PATH}"
        )
    sk_model, _ = load_sklearn_model()
    class SklearnAdapter:
        backend = "mlp"
        def predict_slot(self, image: np.ndarray) -> tuple[str, float]:
            feats = image_to_features(image).reshape(1, -1)
            probs = sk_model.predict_proba(feats)[0]
            idx = int(np.argmax(probs))
            return CLASS_NAMES[idx], float(probs[idx])
    return SklearnAdapter()
def predict_slot(model, image: np.ndarray) -> tuple[str, float]:
    return model.predict_slot(image)
def predict_parking_lot(model, image: np.ndarray, rows: int, cols: int) -> list[dict]:
    slots = split_into_slots(image, rows, cols)
    for slot in slots:
        label, confidence = predict_slot(model, slot["image"])
        slot["label"] = label
        slot["confidence"] = round(confidence, 3)
    return slots
