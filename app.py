import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("animal_footprint_clean.keras")
    return model

model = load_model()

# Load class names (same order as during training)
CLASS_NAMES = ['cat', 'dog', 'deer', 'elephant']  # 🔹 Replace with your actual class names

# App title
st.title("🐾 Animal Footprint Detection")
st.markdown("Upload an image of an animal footprint and let the model identify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Footprint", use_container_width=True)

    # Preprocess image
    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    pred_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Display result
    st.subheader(f"Prediction: **{pred_class.upper()}** 🐾")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show probability bar
    st.bar_chart(dict(zip(CLASS_NAMES, score.numpy())))
