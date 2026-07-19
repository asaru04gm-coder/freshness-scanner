import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Freshness Scanner",
    page_icon="🥬",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

IMG_SIZE = (128, 128)

st.title("🥬 Fruit & Vegetable Freshness Scanner")

st.write(
    "Upload an image of a fruit or vegetable to check whether it is **Fresh** or **Stale**."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)

    img = np.array(img).astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = float(model.predict(img, verbose=0)[0][0])

    if prediction > 0.5:
        label = "🍂 Stale"
        confidence = prediction
        st.error(label)
    else:
        label = "🥬 Fresh"
        confidence = 1 - prediction
        st.success(label)

    st.progress(int(confidence * 100))

    st.write(f"**Confidence:** {confidence*100:.2f}%")
