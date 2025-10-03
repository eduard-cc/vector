import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Food Analyzer",
    page_icon="🍔",
    layout="centered"
)

# --- MODEL AND DATA FUNCTIONS ---

def build_model(num_classes):
    """Builds the model architecture."""
    # Define the model architecture exactly as it was during training
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet', # It's okay to load imagenet weights here, we'll override them
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1./255),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

@st.cache_resource
def load_model_with_weights(weights_path, num_classes):
    """Builds the model and loads the trained weights."""
    try:
        model = build_model(num_classes)
        model.load_weights(weights_path)

        # FIX: Perform a "dummy" prediction to fully initialize the model's state.
        # This is a known workaround for issues with BatchNormalization layers upon loading.
        dummy_input = np.zeros((1, 224, 224, 3))
        model.predict(dummy_input)

        return model
    except Exception as e:
        st.error(f"Error loading model with weights: {e}")
        return None

def load_class_names(class_names_path):
    """Loads the class names from a text file."""
    try:
        with open(class_names_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except FileNotFoundError:
        st.error(f"Error: class_names.txt not found at {class_names_path}")
        return None

def preprocess_image(image):
    """Preprocesses the image for the model."""
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array

def get_prediction(model, class_names, processed_image):
    """Gets a prediction from the model."""
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence

# --- FILE PATHS ---
WEIGHTS_PATH = os.path.join("models", "food_vision_model_quick.weights.h5")
CLASS_NAMES_PATH = "class_names.txt"


# --- MAIN APP ---
st.title("🍔 AI Food Analyzer")
st.markdown("Upload an image of a food item, and the AI will try to identify it!")

# Load the class names
CLASS_NAMES = load_class_names(CLASS_NAMES_PATH)

if CLASS_NAMES is not None:
    # Load the model with the weights
    model = load_model_with_weights(WEIGHTS_PATH, len(CLASS_NAMES))

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence = get_prediction(model, CLASS_NAMES, processed_image)

                st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
                st.info(f"**Confidence:** {confidence:.2f}%")
    elif model is None:
         st.warning(f"Model weights not found. Please make sure `{WEIGHTS_PATH}` exists.")
else:
    st.warning(f"Class names file not found. Please make sure `{CLASS_NAMES_PATH}` exists.")