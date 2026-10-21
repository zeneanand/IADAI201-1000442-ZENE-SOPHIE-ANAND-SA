"""Image resizing, augmentation, and slot-grid cropping."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
TARGET_SIZE = (224, 224)
CLASS_NAMES = ("empty", "occupied")
def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)
def resize_224(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
def to_model_input(image: np.ndarray) -> np.ndarray:
    rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized = resize_224(rgb)
    batch = np.expand_dims(resized.astype("float32"), axis=0)
    return batch
def split_into_slots(image: np.ndarray, rows: int, cols: int) -> list[dict]:
    """Crop a full parking-lot image into a fixed rectangular grid of slots."""
    height, width = image.shape[:2]
    slot_h = height // rows
    slot_w = width // cols
    crops = []
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * slot_h, c * slot_w
            y2 = height if r == rows - 1 else (r + 1) * slot_h
            x2 = width if c == cols - 1 else (c + 1) * slot_w
            crop = image[y1:y2, x1:x2]
            crops.append(
                {
                    "id": r * cols + c + 1,
                    "row": r,
                    "col": c,
                    "box": (x1, y1, x2, y2),
                    "image": crop,
                }
            )
    return crops
def draw_annotations(image: np.ndarray, slots: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for slot in slots:
        x1, y1, x2, y2 = slot["box"]
        occupied = slot["label"] == "occupied"
        color = (239, 68, 68) if occupied else (34, 197, 94)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        caption = f"S{slot['id']}: {'OCC' if occupied else 'FREE'}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1), (x1 + tw + 8, y1 + th + 10), color, -1)
        cv2.putText(
            canvas,
            caption,
            (x1 + 4, y1 + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
