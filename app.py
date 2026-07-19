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


st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8f1;
    }
    .app-shell {
        max-width: 520px;
        margin: 0 auto;
        padding: 10px 0 24px;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: #1e3227;
    }
    .brand {
        font-family: "Georgia", "Times New Roman", serif;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #2a5a3a;
        margin-bottom: 4px;
    }
    .brand span {
        color: #a5602e;
    }
    .tagline {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #4d5d4d;
        margin-bottom: 14px;
    }
    .preview-card {
        border-radius: 24px;
        border: 1px solid rgba(30, 50, 39, 0.14);
        background: linear-gradient(140deg, #2a5a3a, #17281f);
        box-shadow: 0 10px 30px rgba(30, 50, 39, 0.12);
        padding: 16px;
        margin-bottom: 16px;
        overflow: hidden;
    }
    .hero-placeholder {
        min-height: 320px;
        border-radius: 16px;
        border: 1px dashed rgba(255,255,255,0.24);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: rgba(246, 248, 241, 0.8);
        text-align: center;
        padding: 24px;
        gap: 10px;
    }
    .hero-placeholder p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.5;
        color: rgba(246, 248, 241, 0.72);
    }
    .preview-card img {
        border-radius: 16px;
        width: 100%;
        max-height: 360px;
        object-fit: cover;
    }
    .chip {
        margin-bottom: 10px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(20, 32, 25, 0.55);
        color: #fff;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .chip .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #d9a441;
    }
    .controls-row {
        display: flex;
        gap: 10px;
        margin: 8px 0 18px;
        flex-wrap: wrap;
    }
    .controls-row .stButton > button {
        border-radius: 999px;
        border: 1px solid rgba(30, 50, 39, 0.14);
        background: white;
        color: #1e3227;
        padding: 0.6rem 1rem;
    }
    .controls-row .stButton > button:hover {
        border-color: #3e7b4f;
        color: #2a5a3a;
    }
    .result-card {
        border-radius: 20px;
        background: white;
        border: 1px solid rgba(30, 50, 39, 0.12);
        box-shadow: 0 8px 24px rgba(30, 50, 39, 0.08);
        padding: 18px;
        margin-top: 10px;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-title.fresh { color: #2a5a3a; }
    .result-title.stale { color: #7c4520; }
    .result-meta {
        color: #647267;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    .meter {
        height: 8px;
        border-radius: 999px;
        background: #ecefe5;
        overflow: hidden;
        margin: 10px 0;
    }
    .meter > div {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #3e7b4f, #d9a441, #a5602e);
    }
    .footer-note {
        margin-top: 16px;
        font-size: 0.8rem;
        color: #758372;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown('<div class="brand">ripe<span>.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Freshness, on the spot</div>', unsafe_allow_html=True)

if "camera_photo_available" not in st.session_state:
    st.session_state.camera_photo_available = False

source = st.radio("Choose an image source", ["Upload image", "Use camera"], horizontal=True)

st.markdown('<div class="preview-card">', unsafe_allow_html=True)
if source == "Upload image":
    input_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
else:
    if st.button("Retake photo", disabled=not st.session_state.camera_photo_available):
        st.session_state.camera_photo_available = False
        st.rerun()

    input_image = st.camera_input("Take a photo", key="camera_input")
    if input_image is not None:
        st.session_state.camera_photo_available = True

if input_image is None:
    st.markdown(
        '<div class="hero-placeholder"><div class="chip"><span class="dot"></span>Live scan</div><h3 style="margin:0; font-weight:600;">Point your camera at a fruit or vegetable</h3><p>Use the camera or upload a photo to check whether it looks fresh or stale.</p></div>',
        unsafe_allow_html=True,
    )
else:
    display_image = load_pil_image(input_image)
    st.image(display_image, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if input_image is not None:
    prediction = predict_freshness(input_image)
    if prediction > 0.5:
        label = "Past its prime"
        confidence = prediction
        verdict_class = "stale"
        accent = "#7c4520"
    else:
        label = "Looks fresh"
        confidence = 1 - prediction
        verdict_class = "fresh"
        accent = "#2a5a3a"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title {verdict_class}">{label}</div>
            <div class="result-meta">Freshness estimate from the scanner model</div>
            <div class="meter"><div style="width:{int(confidence * 100)}%; background:linear-gradient(90deg, {accent}, #d9a441);"></div></div>
            <div style="font-size:0.9rem; color:#425245;">Confidence: {confidence * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer-note">Camera-ready • Upload or snap a photo to get started</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
