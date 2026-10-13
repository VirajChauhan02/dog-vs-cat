"""Train a compact CNN from Kaggle's labelled cat.N.jpg and dog.N.jpg images."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "train"
MODEL_PATH = ROOT / "model.keras"
IMAGE_SIZE = (128, 128)
SEED = 42
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_labeled_images():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Training folder not found: {DATA_DIR}. Extract Kaggle's labelled "
            "train.zip images into data/train."
        )

    paths = []
    labels = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        name = path.name.lower()
        if name.startswith("cat."):
            paths.append(str(path))
            labels.append(0)
        elif name.startswith("dog."):
            paths.append(str(path))
            labels.append(1)

    labels = np.asarray(labels, dtype=np.int32)
    paths = np.asarray(paths)
    if not len(paths) or set(labels.tolist()) != {0, 1}:
        raise ValueError(
            "No labelled cat.N.* and dog.N.* images were found. Put the "
            "labelled Kaggle training images under data/train."
        )
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 2:
        raise ValueError("At least two labelled images per class are required.")
    return paths, labels


def stratified_split(paths, labels):
    rng = np.random.default_rng(SEED)
    train_indices = []
    validation_indices = []
    for class_id in (0, 1):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * 0.2)))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return (paths[train_indices], labels[train_indices],
            paths[validation_indices], labels[validation_indices])


def make_dataset(paths, labels, training):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        data = tf.io.read_file(path)
        image = tf.io.decode_image(data, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(paths), 4096), seed=SEED)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model():
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    paths, labels = find_labeled_images()
    x_train, y_train, x_val, y_val = stratified_split(paths, labels)
    print(f"Found {len(paths)} images: {np.bincount(labels).tolist()} [cats, dogs].")
    print(f"Training on {len(x_train)} images; validating on {len(x_val)} images.")

    train_ds = make_dataset(x_train, y_train, training=True)
    validation_ds = make_dataset(x_val, y_val, training=False)
    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor="val_loss", save_best_only=True
        ),
    ]
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=12,
        callbacks=callbacks,
    )
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

