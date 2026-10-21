from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["empty", "occupied"]

def load_split(split, shuffle):
    path = DATA_DIR / split

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_dataset.py first."
        )

    return tf.keras.utils.image_dataset_from_directory(
        path,
        class_names=CLASS_NAMES,
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED
    )

train_ds = load_split("train", True)
val_ds = load_split("val", False)
test_ds = load_split("test", False)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.04),
    tf.keras.layers.RandomBrightness(0.15)
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model_path = MODEL_DIR / "parking_model.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
        monitor="val_loss",
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=2,
        factor=0.3
    )
]

print("Training MobileNetV2...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks
)

print("Fine-tuning MobileNetV2...")
base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks
)

model = tf.keras.models.load_model(model_path)

print("\nTest evaluation:")
metrics = model.evaluate(test_ds, return_dict=True)
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

y_true = []
y_prob = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0).ravel()
    y_prob.extend(probabilities)
    y_true.extend(labels.numpy().ravel().astype(int))

y_true = np.array(y_true)
y_pred = (np.array(y_prob) >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion matrix:\n", cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)
display.plot(values_format="d")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=160)
plt.show()

with open("test_metrics.json", "w") as file:
    json.dump(
        {key: float(value) for key, value in metrics.items()},
        file,
        indent=2
    )

print(f"Model saved to: {model_path}")
