"""Train MobileNetV2 on 224×224 slot crops (TensorFlow).
Run this on Google Colab, a GPU machine, or Streamlit Cloud-style Linux
where `import tensorflow` succeeds. Saves models/parking_mobilenet.keras.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
ROOT = Path(__file__).resolve().parents[1]
def make_dataset(folder: Path, image_size: int, batch_size: int, shuffle: bool):
    return keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="categorical",
        class_names=["empty", "occupied"],
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=42,
        shuffle=shuffle,
    )
def build_model(image_size: int) -> keras.Model:
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomBrightness(0.18),
            layers.RandomContrast(0.15),
            layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )
    base = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)
    x = base(x, training=False)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(2, activation="softmax", name="occupancy")(x)
    model = keras.Model(inputs, outputs, name="parking_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    autotune = tf.data.AUTOTUNE
    train_ds = make_dataset(args.splits_dir / "train", args.image_size, args.batch_size, True).prefetch(autotune)
    val_ds = make_dataset(args.splits_dir / "val", args.image_size, args.batch_size, False).prefetch(autotune)
    test_ds = make_dataset(args.splits_dir / "test", args.image_size, args.batch_size, False).prefetch(autotune)
    model = build_model(args.image_size)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                models_dir / "parking_mobilenet.keras",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
            ),
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(models_dir / "training_curves.png", dpi=140)
    best = keras.models.load_model(models_dir / "parking_mobilenet.keras")
    test_loss, test_acc = best.evaluate(test_ds, verbose=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "epochs_ran": len(history.history["loss"]),
        "classes": ["empty", "occupied"],
        "input_size": args.image_size,
        "approach": "crop-based MobileNetV2 classification on a fixed slot grid",
    }
    (models_dir / "mobilenet_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    main()
