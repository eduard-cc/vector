import os
import shutil
import subprocess
import tarfile
import zipfile

import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive  # type: ignore[import-not-found]

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

drive.mount("/content/drive")

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/ai-food-analyzer/"
DRIVE_DATASETS_DIR = "/content/drive/MyDrive/colab_datasets/"
FOOD101_ARCHIVE_PATH = os.path.join(DRIVE_DATASETS_DIR, "food-101.tar.gz")

local_extract_path = "/content/datasets/"
food_dir = os.path.join(local_extract_path, "food-101", "images")
non_food_dir = os.path.join(food_dir, "non_food")

output_model_path = os.path.join(DRIVE_PROJECT_DIR, "model_finetuned.keras")
output_weights_path = os.path.join(DRIVE_PROJECT_DIR, "model_finetuned.weights.h5")
output_plot_path = os.path.join(DRIVE_PROJECT_DIR, "performance_plot_finetuned.png")
class_names_path = os.path.join(DRIVE_PROJECT_DIR, "class_names.txt")

os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
os.makedirs(DRIVE_DATASETS_DIR, exist_ok=True)


def save_performance_plot(history, history_fine_tune, initial_epochs, save_path):
    """Saves the combined training and validation performance plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    acc = history.history["accuracy"] + history_fine_tune.history["accuracy"]
    val_acc = (
        history.history["val_accuracy"] + history_fine_tune.history["val_accuracy"]
    )
    loss = history.history["loss"] + history_fine_tune.history["loss"]
    val_loss = history.history["val_loss"] + history_fine_tune.history["val_loss"]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(
        initial_epochs - 1, linestyle="--", color="k", label="Start Fine-Tuning"
    )
    plt.legend(loc="lower right")
    plt.title("Combined Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(
        initial_epochs - 1, linestyle="--", color="k", label="Start Fine-Tuning"
    )
    plt.legend(loc="upper right")
    plt.title("Combined Loss")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Combined performance plot saved to: {save_path}")


# --- Helper: Find Files/Folders Recursively ---
def find_file(start_dir, filename):
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def find_folder(start_dir, foldername):
    for root, dirs, files in os.walk(start_dir):
        if foldername in dirs:
            return os.path.join(root, foldername)
    return None


def get_data():
    """Handles data download and extraction for Colab."""
    global food_dir

    if not os.path.exists(FOOD101_ARCHIVE_PATH):
        print("Downloading Food-101 dataset to Google Drive...")
        subprocess.run(
            [
                "wget",
                "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "-O",
                FOOD101_ARCHIVE_PATH,
            ],
            check=True,
        )

    print("Copying Food-101 from Drive to Colab runtime...")
    subprocess.run(["cp", FOOD101_ARCHIVE_PATH, "/content/"], check=True)

    print("Extracting Food-101...")
    if os.path.exists(local_extract_path):
        shutil.rmtree(local_extract_path)

    with tarfile.open("/content/food-101.tar.gz", "r:gz") as tar:
        tar.extractall(path=local_extract_path)

    print("Downloading Caltech-101 for 'non_food' class...")
    caltech_url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
    caltech_zip = "/content/caltech101.zip"
    caltech_extract_base = "/content/caltech101_raw"

    if not os.path.exists(caltech_zip):
        subprocess.run(["wget", caltech_url, "-O", caltech_zip], check=True)

    print("Extracting Caltech-101 (Outer Zip)...")
    if os.path.exists(caltech_extract_base):
        shutil.rmtree(caltech_extract_base)

    with zipfile.ZipFile(caltech_zip, "r") as zip_ref:
        zip_ref.extractall(caltech_extract_base)

    print("Extracting Caltech-101 (Inner Tar)...")
    inner_tar_path = find_file(caltech_extract_base, "101_ObjectCategories.tar.gz")
    if not inner_tar_path:
        raise FileNotFoundError(
            "Could not find '101_ObjectCategories.tar.gz' inside Caltech zip."
        )

    tar_dir = os.path.dirname(inner_tar_path)
    with tarfile.open(inner_tar_path, "r:gz") as tar:
        tar.extractall(path=tar_dir)

    base_caltech = find_folder(caltech_extract_base, "101_ObjectCategories")
    if not base_caltech:
        raise FileNotFoundError(
            "Could not find '101_ObjectCategories' folder after extraction."
        )

    # exclude items that are food or similar
    excluded_categories = [
        "BACKGROUND_Google",  # Noise
        "bass",  # Fish
        "brain",  # Looks like meat
        "crab",  # Seafood
        "crayfish",  # Seafood
        "cup",  # Often contains coffee/tea/soup
        "lobster",  # Seafood
        "octopus",  # Seafood
        "pizza",  # Food
        "rooster",  # Poultry
        "strawberry",  # Fruit
        "sunflower",  # Seeds/Oil
    ]

    print(f"Constructing 'non_food' class (excluding: {excluded_categories})...")
    os.makedirs(non_food_dir, exist_ok=True)

    count = 0
    for category in os.listdir(base_caltech):
        if category in excluded_categories:
            print(f"Skipping food-like category: {category}")
            continue

        cat_path = os.path.join(base_caltech, category)
        if not os.path.isdir(cat_path):
            continue

        # Move images to our single 'non_food' directory
        for img_name in os.listdir(cat_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(cat_path, img_name)
                # Rename to avoid collisions (e.g., airplane_001.jpg)
                dst = os.path.join(non_food_dir, f"{category}_{img_name}")
                shutil.move(src, dst)
                count += 1

    print(
        f"Successfully created 'non_food' class with {count} images from Caltech-101."
    )

    # Reset food_dir to point to the main images folder which now includes 'non_food'
    food_dir = os.path.join(local_extract_path, "food-101", "images")
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
    print(f"Total classes: {len(class_names)} (Expected: 102)")

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
    base_model = model.get_layer(index=1)
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
    train_dataset, validation_dataset, class_names = create_data_pipelines(food_dir)

    num_classes = len(class_names)
    print(f"Building model for {num_classes} classes...")

    model = build_model(num_classes)
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
