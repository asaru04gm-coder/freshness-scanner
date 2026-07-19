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
        background: radial-gradient(circle at top, #f6f8f1 0%, #e6ebdc 32%, #dbe0c8 100%) !important;
    }
    .app-shell {
        max-width: 820px;
        margin: 0 auto;
        padding: 24px 20px 32px;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: #1e3227;
    }
    .brand {
        font-family: "Georgia", "Times New Roman", serif;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #2a5a3a;
        margin-bottom: 4px;
    }
    .brand span {
        color: #a5602e;
    }
    .tagline {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #4d5d4d;
        margin-bottom: 20px;
    }
    .hero-card {
        border-radius: 34px;
        padding: 32px 28px;
        margin-bottom: 24px;
        background: linear-gradient(135deg, rgba(35, 71, 42, 0.98), rgba(24, 38, 28, 0.98));
        box-shadow: 0 24px 70px rgba(20, 40, 26, 0.16);
        border: 1px solid rgba(246, 248, 241, 0.12);
        overflow: hidden;
        position: relative;
    }
    .hero-card::before {
        content: '';
        position: absolute;
        left: -40px;
        top: 20px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(217,164,65,0.28), transparent 55%);
        filter: blur(18px);
        pointer-events: none;
    }
    .hero-card::after {
        content: '';
        position: absolute;
        right: -32px;
        bottom: -20px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(255,255,255,0.14), transparent 60%);
        filter: blur(10px);
        pointer-events: none;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #f6f8f1;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 18px;
        width: fit-content;
    }
    .hero-tag .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #d9a441;
        box-shadow: 0 0 10px rgba(217,164,65,0.8);
    }
    .hero-title {
        font-size: clamp(2.6rem, 6vw, 4.2rem);
        line-height: 0.92;
        font-weight: 900;
        margin: 0;
        color: #fff;
        letter-spacing: -0.04em;
        text-shadow: 0 18px 30px rgba(0,0,0,0.18);
    }
    .hero-subtitle {
        margin: 18px 0 0;
        font-size: 1rem;
        color: rgba(255,255,255,0.88);
        max-width: 640px;
        line-height: 1.7;
    }
    .hero-wave {
        margin-top: 32px;
        height: 2px;
        width: 100%;
        background: linear-gradient(90deg, rgba(217,164,65,0), rgba(217,164,65,0.7), rgba(217,164,65,0));
        position: relative;
        overflow: hidden;
    }
    .hero-wave::before {
        content: '';
        position: absolute;
        top: -8px;
        left: -60px;
        width: 120px;
        height: 24px;
        background: rgba(217,164,65,0.35);
        border-radius: 999px;
        filter: blur(8px);
        animation: wave-glow 3.6s ease-in-out infinite;
    }
    @keyframes wave-glow {
        0%, 100% { transform: translateX(0); }
        50% { transform: translateX(620px); }
    }
    .preview-card {
        border-radius: 32px;
        border: 1px solid rgba(249, 250, 244, 0.16);
        background: rgba(249, 250, 244, 0.95);
        box-shadow: 0 24px 60px rgba(30, 50, 39, 0.08);
        padding: 20px;
        margin-bottom: 22px;
        overflow: hidden;
        position: relative;
    }
    .preview-card::before {
        content: '';
        position: absolute;
        top: 18px;
        right: 18px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(58,122,76,0.08), transparent 60%);
        pointer-events: none;
    }
    .hero-placeholder {
        min-height: 340px;
        border-radius: 24px;
        border: 1px dashed rgba(30, 50, 39, 0.16);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #2b3d2b;
        text-align: center;
        padding: 44px 32px;
        gap: 16px;
        background: rgba(255,255,255,0.85);
    }
    .hero-placeholder h3 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e3227;
    }
    .hero-placeholder p {
        margin: 0;
        font-size: 1rem;
        line-height: 1.7;
        color: #4b5c47;
        max-width: 520px;
    }
    .hero-placeholder svg {
        width: 46px;
        height: 46px;
        color: #2a5a3a;
        opacity: 0.9;
    }
    .preview-card img {
        border-radius: 24px;
        width: 100%;
        max-height: 420px;
        object-fit: cover;
        border: 1px solid rgba(30, 50, 39, 0.08);
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 14px;
        border-radius: 999px;
        background: rgba(58, 122, 76, 0.12);
        color: #2a5a3a;
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .chip .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #d9a441;
        box-shadow: 0 0 12px rgba(217,164,65,0.6);
    }
    .stRadio label {
        color: #1e3227 !important;
        font-weight: 700 !important;
    }
    .stRadio [role="radio"] {
        accent-color: #a5602e !important;
    }
    .stRadio .stRadio > div,
    .stRadio .css-1l02ky3 {
        color: #1e3227 !important;
    }
    .controls-row {
        display: flex;
        gap: 12px;
        margin: 14px 0 24px;
        flex-wrap: wrap;
        justify-content: center;
    }
    .stButton > button {
        border-radius: 999px !important;
        border: none !important;
        background: linear-gradient(135deg, #3e7b4f, #a5602e) !important;
        color: #fff !important;
        padding: 0.88rem 1.6rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 16px 35px rgba(30, 50, 39, 0.18) !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 18px 40px rgba(30, 50, 39, 0.2) !important;
    }
    .result-card {
        border-radius: 28px;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(30, 50, 39, 0.1);
        box-shadow: 0 18px 50px rgba(30, 50, 39, 0.08);
        padding: 24px;
        margin-top: 6px;
    }
    .result-title {
        font-size: 1.7rem;
        font-weight: 900;
        margin-bottom: 6px;
    }
    .result-title.fresh { color: #2a5a3a; }
    .result-title.stale { color: #7c4520; }
    .result-meta {
        color: #556353;
        font-size: 0.95rem;
        margin-bottom: 18px;
    }
    .meter {
        height: 10px;
        border-radius: 999px;
        background: #ebece4;
        overflow: hidden;
        margin: 14px 0 12px;
    }
    .meter > div {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #3e7b4f, #d9a441, #a5602e);
    }
    .footer-note {
        margin-top: 18px;
        font-size: 0.82rem;
        color: #66715c;
        text-align: center;
    }
    .stFileUploader {
        border-radius: 22px !important;
        background: rgba(30, 50, 39, 0.08) !important;
        border: 1px dashed rgba(30, 50, 39, 0.18) !important;
    }
    .stFileUploader label {
        width: 100% !important;
        color: #1e3227 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown('<div class="hero-card"><div class="hero-tag"><span class="dot"></span>Live freshness scanner</div><h1 class="hero-title">ripe<span>.</span></h1><p class="hero-subtitle">Capture food with your camera, upload a photo, and get an instant freshness verdict so you can eat smarter and waste less.</p><div class="hero-wave"></div></div>', unsafe_allow_html=True)

if "camera_photo_available" not in st.session_state:
    st.session_state.camera_photo_available = False

source = st.radio("Choose an image source", ["Upload image", "Use camera"], horizontal=True, label_visibility="visible")

st.markdown('<div class="preview-card">', unsafe_allow_html=True)
if source == "Upload image":
    input_image = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], label_visibility="visible")
else:
    if st.button("Retake photo", disabled=not st.session_state.camera_photo_available):
        st.session_state.camera_photo_available = False
        st.rerun()

    input_image = st.camera_input("Capture a live photo", key="camera_input")
    if input_image is not None:
        st.session_state.camera_photo_available = True

if input_image is None:
    st.markdown(
        '<div class="hero-placeholder"><svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 8C4 6.89543 4.89543 6 6 6H7.5L8.5 4H15.5L16.5 6H18C19.1046 6 20 6.89543 20 8V17C20 18.1046 19.1046 19 18 19H6C4.89543 19 4 18.1046 4 17V8Z" stroke="currentColor" stroke-width="1.4"/><circle cx="12" cy="12.5" r="3.6" stroke="currentColor" stroke-width="1.4"/></svg><div class="chip"><span class="dot"></span>Live scan</div><h3>Point your camera at a fruit or vegetable</h3><p>Use the camera or upload a photo to check whether it looks fresh or stale.</p></div>',
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
            <div style="font-size:0.95rem; color:#425245;">Confidence: {confidence * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer-note">Camera-ready • Upload or snap a photo to get started</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
