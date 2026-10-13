"""Streamlit interface for the trained cat-versus-dog classifier."""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.keras"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ("Cat", "Dog")


st.set_page_config(page_title="Cat and Dog Classifier", page_icon="🐾")
st.title("Cat and Dog Image Classifier")
st.write("Upload an image to classify it as a cat or a dog.")


@st.cache_resource
def load_classifier():
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH, compile=False)


if not MODEL_PATH.is_file():
    st.error("The trained model is not available yet.")
    st.info(
        "Place the labelled Kaggle training images in data/train, then run "
        "start.ps1. It will train model.keras once and start this app."
    )
    st.stop()

try:
    model = load_classifier()
except Exception as exc:  # Present an actionable message for a missing or invalid model.
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image", type=("jpg", "jpeg", "png", "webp")
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image to classify", use_container_width=True)
        if st.button("Classify image", type="primary"):
            resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            batch = np.asarray(resized, dtype=np.float32)[None, ...] / 255.0
            probabilities = np.asarray(model.predict(batch, verbose=0))[0]
            if probabilities.size != len(CLASS_NAMES):
                st.error("The model output does not match the expected Cat/Dog classes.")
            else:
                index = int(np.argmax(probabilities))
                st.success(f"Prediction: **{CLASS_NAMES[index]}**")
                st.caption(f"Model confidence: {probabilities[index]:.1%}")
    except Exception as exc:
        st.error(f"This image could not be processed: {exc}")

