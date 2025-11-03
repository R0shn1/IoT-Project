import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Animal Footprint Tracker 🐾",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM STYLES ----------
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
            padding: 2rem;
            border-radius: 15px;
        }
        h1 {
            color: #4a4a8c;
            text-align: center;
            font-family: 'Poppins', sans-serif;
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 1.1em;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #357ABD;
        }
        .uploadedImage {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_footprint_optimized_v9_refined.keras")
    return model

model = load_model()

# ---------- CLASS NAMES ----------
CLASS_NAMES = ['canid', 'cervid', 'felid', 'mustelid', 'ursid']

# ---------- SIDEBAR ----------
st.sidebar.header("About 🦴")
st.sidebar.info("""
Upload an image of an **animal footprint**, and this model will predict which **family** it belongs to.

**Categories**:
- 🐕 Canid (Dogs, Foxes)
- 🦌 Cervid (Deer)
- 🐈 Felid (Cats)
- 🦦 Mustelid (Otters, Weasels)
- 🐻 Ursid (Bears)
""")

# ---------- MAIN APP ----------
st.title("🐾 Animal Footprint Detection")
st.markdown("### Upload a footprint image and let AI identify the animal family!")

uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div class='uploadedImage'>", unsafe_allow_html=True)
    st.image(image, caption="Uploaded Footprint", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    pred_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"### 🐾 Prediction: **{pred_class.upper()}**")
    st.progress(int(confidence))
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Probability bar
    st.markdown("#### 📊 Class Probabilities")
    st.bar_chart(dict(zip(CLASS_NAMES, score.numpy())))
else:
    st.info("⬆️ Upload an image to start the prediction.")
