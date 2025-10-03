import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

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

@st.cache_data
def fetch_nutrition_data(food_name):
    """Fetches nutritional data from the Open Food Facts API."""
    # Format the food name for the API (e.g., "apple_pie" -> "apple pie")
    search_term = food_name.replace('_', ' ')
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={search_term}&search_simple=1&action=process&json=1"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        if data['products']:
            # Get the first product's nutritional info
            product = data['products'][0]
            nutriments = product.get('nutriments', {})
            # Get values per 100g, with fallbacks to 0
            return {
                "calories": nutriments.get('energy-kcal_100g', 0),
                "protein": nutriments.get('proteins_100g', 0),
                "fat": nutriments.get('fat_100g', 0),
                "carbs": nutriments.get('carbohydrates_100g', 0)
            }
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

# --- FILE PATHS ---
WEIGHTS_PATH = os.path.join("models", "food_vision_model_quick.weights.h5")
CLASS_NAMES_PATH = "class_names.txt"


# --- MAIN APP ---
st.title("🍔 AI Food Analyzer")
st.markdown("Upload an image of a food item, and the AI will try to identify it and fetch its nutritional data.")

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
            with st.spinner("1/2 - Identifying food..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence = get_prediction(model, CLASS_NAMES, processed_image)

            st.success(f"**Prediction:** {predicted_class.replace('_', ' ').title()} ({confidence:.2f}%)")

            with st.spinner("2/2 - Fetching nutritional data..."):
                nutrition_data = fetch_nutrition_data(predicted_class)

            if nutrition_data:
                st.subheader("Estimated Nutritional Information (per 100g)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Calories", f"{nutrition_data['calories']:.0f} kcal")
                col2.metric("Protein", f"{nutrition_data['protein']:.1f} g")
                col3.metric("Fat", f"{nutrition_data['fat']:.1f} g")
                col4.metric("Carbs", f"{nutrition_data['carbs']:.1f} g")
            else:
                st.warning("Could not retrieve nutritional information for this food.")

else:
    st.warning(f"Required files not found. Please ensure `{CLASS_NAMES_PATH}` and `{WEIGHTS_PATH}` are in the correct locations.")

