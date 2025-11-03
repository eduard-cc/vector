import os
import shutil
import subprocess
import tarfile

import tensorflow as tf
from google.colab import drive  # type: ignore[import-not-found]

from plot import save_performance_plot

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 101
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

drive.mount("/content/drive")

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/ai-food-analyzer/"
DRIVE_DATASETS_DIR = "/content/drive/MyDrive/colab_datasets/"
FOOD101_ARCHIVE_PATH = os.path.join(DRIVE_DATASETS_DIR, "food-101.tar.gz")

local_extract_path = "/content/datasets/"
food_dir = os.path.join(local_extract_path, "food-101", "images")

output_model_path = os.path.join(DRIVE_PROJECT_DIR, "model_finetuned.keras")
output_weights_path = os.path.join(DRIVE_PROJECT_DIR, "model_finetuned.weights.h5")
output_plot_path = os.path.join(DRIVE_PROJECT_DIR, "performance_plot_finetuned.png")
class_names_path = os.path.join(DRIVE_PROJECT_DIR, "class_names.txt")

os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
os.makedirs(DRIVE_DATASETS_DIR, exist_ok=True)


def get_data():
    """Handles data download and extraction for Colab."""
    global food_dir

    if not os.path.exists(FOOD101_ARCHIVE_PATH):
        print("Downloading dataset to Google Drive...")
        subprocess.run(
            [
                "wget",
                "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "-O",
                FOOD101_ARCHIVE_PATH,
            ],
            check=True,
        )

    print("Copying dataset from Drive to Colab runtime...")
    subprocess.run(["cp", FOOD101_ARCHIVE_PATH, "/content/"], check=True)

    print("Extracting dataset...")
    if os.path.exists(local_extract_path):
        shutil.rmtree(local_extract_path)

    with tarfile.open("/content/food-101.tar.gz", "r:gz") as tar:
        tar.extractall(path=local_extract_path)

    food_dir = os.path.join(local_extract_path, "food-101", "images")
    print(f"Data is ready. Using image directory: {food_dir}")
    return food_dir


def create_data_pipelines(food_dir):
    """Creates and optimizes tf.data.Dataset pipelines."""
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        food_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        food_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_dataset.class_names
    with open(class_names_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to {class_names_path}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, class_names


def build_model(num_classes):
    """Builds the initial, frozen-base model."""
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMG_SIZE + (3,)),
            tf.keras.layers.Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def fine_tune_model(model):
    """Unfreezes and re-compiles the model for fine-tuning."""
    base_model = model.get_layer(index=2)
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )
    return model


def main():
    print("Running in Google Colab mode.")

    food_dir = get_data()
    train_dataset, validation_dataset, _ = create_data_pipelines(food_dir)

    model = build_model(NUM_CLASSES)
    model.summary()

    print(f"\n--- Starting initial training for {INITIAL_EPOCHS} epochs... ---")
    history = model.fit(
        train_dataset, validation_data=validation_dataset, epochs=INITIAL_EPOCHS
    )

    print("\n--- Starting Fine-Tuning ---")
    model = fine_tune_model(model)
    model.summary()

    print(f"\n--- Starting fine-tuning for {FINE_TUNE_EPOCHS} more epochs... ---")
    history_fine_tune = model.fit(
        train_dataset,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
    )

    print("\n--- Training complete. Saving artifacts... ---")

    save_performance_plot(history, history_fine_tune, INITIAL_EPOCHS, output_plot_path)

    model.save(output_model_path)
    model.save_weights(output_weights_path)
    print(f"Final model and weights saved to {output_weights_path}")
    print("\n--- Pipeline finished successfully! ---")


if __name__ == "__main__":
    main()
