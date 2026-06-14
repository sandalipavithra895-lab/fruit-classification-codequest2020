import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Smart Waste Classifier", layout="centered")

st.title("♻️ Smart Waste Classification System")
st.write("FCIT - Code Quest 2026 AI Project")

# Model එක Load කිරීම
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('waste_classifier_model.h5')

model = load_my_model()

# Class Names Load කිරීම
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Image Uploading
uploaded_file = st.file_uploader("රූපයක් Upload කරන්න...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("🔄 පිරික්සමින් පවතී...")
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Prediction
    predictions = model.predict(img_array)
    highest_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[highest_class_idx]
    confidence = predictions[0][highest_class_idx] * 100
    
    # ප්‍රතිඵල පෙන්වීම
    st.success(f"**Predicted Category:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    
    st.subheader("📊 Class Probabilities")
    for i, name in enumerate(class_names):
        prob = predictions[0][i] * 100
        st.write(f"{name}: {prob:.2f}%")
