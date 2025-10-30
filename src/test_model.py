import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tarfile
import numpy as np
import subprocess
import shutil

# --- 1. ENVIRONMENT DETECTION & PARAMETERS ---

IN_COLAB = 'COLAB_GPU' in os.environ

# Model Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 101
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# --- 2. SET UP PATHS ---

if IN_COLAB:
    from google.colab import drive # type: ignore[import-not-found]
    drive.mount('/content/drive')

    DRIVE_PROJECT_DIR = "/content/drive/MyDrive/ai-food-analyzer/"
    DRIVE_DATASETS_DIR = "/content/drive/MyDrive/colab_datasets/"
    FOOD101_ARCHIVE_PATH = os.path.join(DRIVE_DATASETS_DIR, "food-101.tar.gz")

    local_extract_path = "/content/datasets/"
    food_dir = os.path.join(local_extract_path, 'food-101', 'images')

    output_model_path = os.path.join(DRIVE_PROJECT_DIR, "food_vision_model_finetuned.keras")
    output_weights_path = os.path.join(DRIVE_PROJECT_DIR, "food_vision_model_finetuned.weights.h5")
    output_plot_path = os.path.join(DRIVE_PROJECT_DIR, "performance_plot_finetuned.png")
    class_names_path = os.path.join(DRIVE_PROJECT_DIR, "class_names.txt")

    os.makedirs(DRIVE_PROJECT_DIR, exist_ok=True)
    os.makedirs(DRIVE_DATASETS_DIR, exist_ok=True)

else:
    home_dir = os.path.expanduser("~")
    data_dir_root = os.path.join(home_dir, '.keras', 'datasets')

    output_model_path = os.path.join("models", "food_vision_model_finetuned.keras")
    output_weights_path = os.path.join("models", "food_vision_model_finetuned.weights.h5")
    output_plot_path = os.path.join("reports", "performance_plot_finetuned.png")
    class_names_path = "class_names.txt"

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

# --- 3. DATA PREPARATION ---

def get_data():
    """Handles data download and extraction for both environments."""
    global food_dir

    if IN_COLAB:
        if not os.path.exists(FOOD101_ARCHIVE_PATH):
            print(f"Downloading dataset to Google Drive. This will happen once.")
            subprocess.run([
                "wget",
                "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "-O", FOOD101_ARCHIVE_PATH
            ], check=True)

        print("Copying dataset from Drive to Colab runtime...")
        subprocess.run(["cp", FOOD101_ARCHIVE_PATH, "/content/"], check=True)

        print("Extracting dataset...")
        if os.path.exists(local_extract_path):
            shutil.rmtree(local_extract_path)

        with tarfile.open("/content/food-101.tar.gz", "r:gz") as tar:
            tar.extractall(path=local_extract_path)

        food_dir = os.path.join(local_extract_path, 'food-101', 'images')

    else:
        print("Downloading or verifying local dataset...")
        dataset_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        data_dir = tf.keras.utils.get_file('food-101', origin=dataset_url, untar=True, cache_dir=data_dir_root)
        food_dir = os.path.join(data_dir, 'food-101', 'images')

    print(f"✅ Data is ready. Using image directory: {food_dir}")
    return food_dir

def create_data_pipelines(food_dir):
    """Creates and optimizes tf.data.Dataset pipelines."""
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        food_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        food_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    with open(class_names_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"✅ Class names saved to {class_names_path}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, validation_dataset, class_names

# --- 4. MODEL BUILDING ---

def build_model(num_classes):
    """Builds the initial, frozen-base model."""
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMG_SIZE + (3,)),
        tf.keras.layers.Rescaling(1./255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
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
        metrics=["accuracy"]
    )
    return model

# --- 5. PLOTTING ---

def save_performance_plot(history, history_fine_tune):
    """Saves the combined training and validation performance plot."""
    acc = history.history['accuracy'] + history_fine_tune.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine_tune.history['val_accuracy']
    loss = history.history['loss'] + history_fine_tune.history['loss']
    val_loss = history.history['val_loss'] + history_fine_tune.history['val_loss']

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(INITIAL_EPOCHS - 1, linestyle='--', color='k', label='Start Fine-Tuning')
    plt.legend(loc='lower right')
    plt.title('Combined Training and Validation Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(INITIAL_EPOCHS - 1, linestyle='--', color='k', label='Start Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title('Combined Training and Validation Loss')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"✅ Combined performance plot saved to: {output_plot_path}")

# --- 6. MAIN EXECUTION ---

def main():
    if not IN_COLAB:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} local GPU(s).")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        else:
            print("⚠️ No local GPU found. Training will be VERY slow on CPU.")

    food_dir = get_data()
    train_dataset, validation_dataset, _ = create_data_pipelines(food_dir)

    model = build_model(NUM_CLASSES)
    model.summary()

    print(f"\n--- Starting initial training for {INITIAL_EPOCHS} epochs... ---")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=INITIAL_EPOCHS
    )

    print("\n--- Starting Fine-Tuning ---")
    model = fine_tune_model(model)
    model.summary()

    print(f"\n--- Starting fine-tuning for {FINE_TUNE_EPOCHS} more epochs... ---")
    history_fine_tune = model.fit(
        train_dataset,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset
    )

    print("\n--- Training complete. Saving artifacts... ---")
    save_performance_plot(history, history_fine_tune)

    model.save(output_model_path)
    model.save_weights(output_weights_path)
    print(f"✅ Final model and weights saved to {output_weights_path}")
    print("\n--- Pipeline finished successfully! ---")

if __name__ == "__main__":
    main()

