import os

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image


def build_model(num_classes):
    """Builds the model architecture."""
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    base_model.trainable = True

    # freeze all layers except for the top 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # create the sequential model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


@st.cache_resource
def load_model_with_weights(weights_path, num_classes):
    """Builds the model and loads the trained weights."""
    try:
        model = build_model(num_classes)

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.load_weights(weights_path)

        # perform a dummy prediction to init the model's state
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
    img_array = tf.expand_dims(img_array, 0)
    return img_array


def get_prediction(model, class_names, processed_image):
    """Gets a prediction from the model and prints debug info."""
    predictions = model.predict(processed_image)

    score = predictions[0]

    top_5_indices = np.argsort(score)[-5:][::-1]
    top_5_scores = score[top_5_indices]
    top_5_classes = [class_names[i] for i in top_5_indices]

    with st.expander("View Debug/Model Probabilities"):
        st.write("Top 5 Predictions:")
        for i in range(5):
            formatted_class = top_5_classes[i].replace("_", " ").title()
            st.write(f"{i + 1}. {formatted_class}: {top_5_scores[i] * 100:.2f}%")

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    return predicted_class, confidence


@st.cache_data
def fetch_nutrition_data(food_name):
    """Fetches nutritional data from the Open Food Facts API."""
    search_term = food_name.replace("_", " ")
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={search_term}&search_simple=1&action=process&json=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data["products"]:
            return None

        product = data["products"][0]
        nutriments = product.get("nutriments", {})

        return {
            "product_name": product.get("product_name", "Generic Match"),
            "calories": nutriments.get("energy-kcal_100g", 0),
            "protein": nutriments.get("proteins_100g", 0),
            "fat": nutriments.get("fat_100g", 0),
            "carbs": nutriments.get("carbohydrates_100g", 0),
        }
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def main():
    st.set_page_config(page_title="AI Food Analyzer", page_icon="🍔", layout="centered")

    st.title("🍔 AI Food Analyzer")

    WEIGHTS_PATH = os.path.join("models", "model_finetuned.weights.h5")
    CLASS_NAMES_PATH = "class_names.txt"

    CLASS_NAMES = load_class_names(CLASS_NAMES_PATH)
    if CLASS_NAMES is None:
        st.stop()

    with st.sidebar:
        st.header("About")
        st.write("This model is trained on the Food-101 dataset.")
        with st.expander("See Supported Foods"):
            st.write(", ".join([c.replace("_", " ").title() for c in CLASS_NAMES]))
        st.caption("Note: The model may perform poorly on foods outside this list, particularly non-Western cuisines.")

    model = load_model_with_weights(WEIGHTS_PATH, len(CLASS_NAMES))
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.stop()

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("1/2 - Identifying food..."):
            processed_image = preprocess_image(image)
            predicted_class, confidence = get_prediction(
                model, CLASS_NAMES, processed_image
            )

        formatted_class = predicted_class.replace("_", " ").title()

        if confidence < 50.0:
            st.warning(f"⚠️ **Low Confidence Prediction** ({confidence:.2f}%)")
            st.write(f"The model thinks this is **{formatted_class}**, but isn't sure.")
            st.write("Nutritional data is hidden to prevent misinformation.")
        else:
            st.success(f"**Prediction:** {formatted_class} ({confidence:.2f}%)")

            with st.spinner("2/2 - Fetching nutritional data..."):
                nutrition_data = fetch_nutrition_data(predicted_class)

            if not nutrition_data:
                st.warning("Could not retrieve nutritional information for this food.")
                return

            st.subheader("Estimated Nutritional Information (per 100g)")

            st.caption(f"Data Source: Open Food Facts (Best match: *{nutrition_data['product_name']}*)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Calories", f"{nutrition_data['calories']:.0f} kcal")
            col2.metric("Protein", f"{nutrition_data['protein']:.1f} g")
            col3.metric("Fat", f"{nutrition_data['fat']:.1f} g")
            col4.metric("Carbs", f"{nutrition_data['carbs']:.1f} g")


if __name__ == "__main__":
    main()
