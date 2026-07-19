import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Freshness Scanner",
    page_icon="🥬",
    layout="centered",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")


@st.cache_resource
def get_model():
    return load_model()


IMG_SIZE = (128, 128)


def load_pil_image(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def preprocess_image(image):
    img = load_pil_image(image).resize(IMG_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    return img_array


def predict_freshness(image):
    model = get_model()
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)
    prediction = float(model.predict(img, verbose=0)[0][0])
    return prediction


st.title("🥬 Fruit & Vegetable Freshness Scanner")

st.write(
    "Upload an image or use your camera to check whether a fruit or vegetable is **Fresh** or **Stale**."
)

if "camera_photo_available" not in st.session_state:
    st.session_state.camera_photo_available = False

source = st.radio("Choose an image source", ["Upload image", "Use camera"], horizontal=True)

if source == "Upload image":
    input_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    if st.button("Retake photo", disabled=not st.session_state.camera_photo_available):
        st.session_state.camera_photo_available = False
        st.rerun()

    input_image = st.camera_input("Take a photo", key="camera_input")
    if input_image is not None:
        st.session_state.camera_photo_available = True

if input_image is not None:
    display_image = load_pil_image(input_image)
    caption = "Captured Image" if source == "Use camera" else "Uploaded Image"
    st.image(display_image, caption=caption, use_container_width=True)

    prediction = predict_freshness(input_image)

    if prediction > 0.5:
        label = "🍂 Stale"
        confidence = prediction
        st.error(label)
    else:
        label = "🥬 Fresh"
        confidence = 1 - prediction
        st.success(label)

    st.progress(int(confidence * 100))
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
