"""ParkVision AI — OBB Training & Dataset Preparation Pipeline.

Prepares and trains YOLOv8-OBB on diagonal, herringbone, and straight parking
stalls using oriented bounding boxes (4-point polygon format: class x1 y1 x2 y2 x3 y3 x4 y4).
"""

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from ultralytics import YOLO

# -------------------------------------------------------------------------
# 1. Dataset Directory Configuration
# -------------------------------------------------------------------------
BASE_DIR = Path.cwd()
DATASET_DIR = BASE_DIR / "dataset_obb"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# -------------------------------------------------------------------------
# 2. Annotation Converters (Handles PKLot rotated XML & Straight/Slanted Polygons)
# -------------------------------------------------------------------------
def convert_pklot_xml_to_obb(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Parse PKLot XML annotations with rotated bounding boxes into YOLO-OBB format.
    
    Extracts either the 4-point contour polygon or converts center, size, and angle
    into normalized 4-point OBB: class x1 y1 x2 y2 x3 y3 x4 y4.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []

    for space in root.findall(".//space"):
        occupied = space.get("occupied", "0")
        class_id = 1 if occupied in ("1", "true", "occupied") else 0

        contour = space.find("contour")
        if contour is not None and len(contour.findall("point")) == 4:
            points = []
            for pt in contour.findall("point"):
                px = float(pt.get("x")) / img_w
                py = float(pt.get("y")) / img_h
                # Clamp coordinates to [0.0, 1.0]
                px = max(0.0, min(1.0, px))
                py = max(0.0, min(1.0, py))
                points.extend([f"{px:.6f}", f"{py:.6f}"])
            
            yolo_lines.append(f"{class_id} " + " ".join(points))
        else:
            # Fallback for rotated rect parameters (center x, y, width, height, angle)
            rotated_rect = space.find("rotatedRect")
            if rotated_rect is not None:
                import cv2
                import numpy as np

                center = rotated_rect.find("center")
                size = rotated_rect.find("size")
                angle = float(rotated_rect.find("angle").get("d", 0))

                cx = float(center.get("x"))
                cy = float(center.get("y"))
                w = float(size.get("w"))
                h = float(size.get("h"))

                # OpenCV RotatedRect to 4 corners
                r_rect = ((cx, cy), (w, h), angle)
                box = cv2.boxPoints(r_rect)

                coords = []
                for pt in box:
                    nx = max(0.0, min(1.0, float(pt[0]) / img_w))
                    ny = max(0.0, min(1.0, float(pt[1]) / img_h))
                    coords.extend([f"{nx:.6f}", f"{ny:.6f}"])

                yolo_lines.append(f"{class_id} " + " ".join(coords))

    return yolo_lines


def build_dataset_yaml(output_dir: Path) -> Path:
    """Generate YOLOv8-OBB dataset YAML configuration."""
    yaml_path = output_dir / "parking_obb.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (output_dir / "images" / "test").exists() else "images/val",
        "names": {
            0: "empty",
            1: "occupied"
        }
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    
    print(f"[INFO] Dataset config written to {yaml_path}")
    return yaml_path


# -------------------------------------------------------------------------
# 3. Training Execution Function
# -------------------------------------------------------------------------
def train_parking_obb(
    data_yaml: Path,
    epochs: int = 60,
    imgsz: int = 960,
    batch_size: int = 8,
    device: str = "0",  # Use "cpu" if no CUDA GPU is present
    export_dir: Path = BASE_DIR / "models"
):
    """Train YOLOv8n-OBB on straight and diagonal parking stalls."""
    print("[INFO] Initializing YOLOv8n-OBB model weights...")
    model = YOLO("yolov8n-obb.pt")

    # Train with augmentations tailored for parking layouts:
    # - degrees=15.0: accounts for varied camera angles
    # - fliplr=0.5: accounts for mirrored rows / dual-lane lots
    # - shear=0.0: strictly disabled to keep diagonal and straight line angles intact
    # - mosaic=1.0: handles dense parking grid features
    print("[INFO] Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=4,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.1,
        degrees=15.0,
        shear=0.0,
        fliplr=0.5,
        flipud=0.0,
        save=True,
        save_period=5,
        project="runs_parkvision",
        name="parkvision_obb",
        exist_ok=True,
    )

    # Validate trained model
    print("[INFO] Running validation...")
    val_results = model.val()

    # Export best weights to models directory for the Streamlit app
    export_dir.mkdir(parents=True, exist_ok=True)
    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    target_pt = export_dir / "parking-yolov8n-obb.pt"

    if best_pt.exists():
        shutil.copy(best_pt, target_pt)
        print(f"[SUCCESS] Trained OBB weights exported directly to: {target_pt}")
    else:
        print(f"[WARNING] Could not locate best.pt at {best_pt}")

    return results


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure expected folder structure exists
    for split in ["train", "val"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    # 1. Create dataset configuration file
    yaml_config = build_dataset_yaml(DATASET_DIR)

    # 2. Check if dataset has training images before launching
    train_images = list((IMAGES_DIR / "train").glob("*.*"))
    if not train_images:
        print("\n" + "=" * 65)
        print("DATASET PLACEHOLDER READY:")
        print(f"1. Place your images in:  {IMAGES_DIR / 'train'}")
        print(f"                          {IMAGES_DIR / 'val'}")
        print(f"2. Place label files in:  {LABELS_DIR / 'train'}")
        print(f"                          {LABELS_DIR / 'val'}")
        print("   Each .txt label format: <class_id> x1 y1 x2 y2 x3 y3 x4 y4")
        print("   (0 = empty, 1 = occupied)")
        print("3. Re-run this script to start training.")
        print("=" * 65)
    else:
        # 3. Execute training
        train_parking_obb(
            data_yaml=yaml_config,
            epochs=60,
            imgsz=960,
            batch_size=8,
            device="0"  # Set to "cpu" if running without GPU
        )
