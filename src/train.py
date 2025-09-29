import tensorflow as tf
import os
import matplotlib.pyplot as plt

# --- 1. SET UP PARAMETERS ---

# Model & Data Parameters
IMG_SIZE = (224, 224)
# REDUCED BATCH SIZE to lower memory usage on laptops
BATCH_SIZE = 16
EPOCHS = 5  # Start with a small number of epochs to test the pipeline
NUM_CLASSES = 101 # For the Food-101 dataset

# Directory Paths
# Programmatically find the path to the dataset
home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, '.keras', 'datasets', 'food-101')
food_dir = os.path.join(data_dir, 'food-101', 'images')

# Define where to save the final model and performance plots
output_model_path = os.path.join("models", "food_vision_model.keras")
output_plot_path = os.path.join("reports", "performance_plot.png")

# Create output directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def main():
    """
    Main function to run the ML pipeline.
    """
    print("--- Starting the Food Vision training pipeline ---")

    # --- 2. CREATE THE DATA PIPELINE ---
    print(f"Loading images from: {food_dir}")
    print("Creating training and validation datasets...")

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

    # Optimize datasets for performance
    # NOTE: .cache() was removed to reduce memory usage on systems with limited RAM.
    # This will make training slower but safer.
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    print("✅ Data pipelines created and optimized.")

    # --- 3. BUILD THE MODEL USING TRANSFER LEARNING (SEQUENTIAL API) ---
    print("Building model with MobileNetV2 base...")

    # 1. Create the base model with a defined input shape
    # Switched to MobileNetV2 for better stability
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=IMG_SIZE + (3,) # Explicitly define the input shape
    )
    base_model.trainable = False # Freeze the base model

    # 2. Create the new model by stacking layers sequentially
    model = tf.keras.Sequential([
        # The input layer matching the data
        tf.keras.layers.Input(shape=IMG_SIZE + (3,)),

        # Add an explicit rescaling layer to normalize pixel values
        tf.keras.layers.Rescaling(1./255),

        # The pre-trained base model for feature extraction
        base_model,

        # The new classifier head
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    print("✅ Model built successfully.")

    # --- 4. COMPILE THE MODEL ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print("✅ Model compiled.")
    model.summary()

    # --- 5. TRAIN THE MODEL ---
    print(f"\n--- Starting model training for {EPOCHS} epochs ---")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )
    print("✅ Model training complete.")

    # --- 6. VISUALIZE PERFORMANCE ---
    print("Generating performance plot...")
    acc = history.history['accuracy']
    val_acc = history.history['validation_accuracy']
    loss = history.history['loss']
    val_loss = history.history['validation_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.savefig(output_plot_path)
    print(f"✅ Performance plot saved to {output_plot_path}")

    # --- 7. SAVE THE TRAINED MODEL ---
    model.save(output_model_path)
    print(f"✅ Final model saved to {output_model_path}")
    print("\n--- Pipeline finished successfully! ---")


if __name__ == "__main__":
    main()

