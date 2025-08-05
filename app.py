import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import base64

# ---------------------------
# Load TensorFlow Model
# ---------------------------
model = tf.keras.models.load_model("plant_disease_model.h5")

class_names = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Corn Cercospora", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy"
]

# ---------------------------
# Prediction Function
# ---------------------------
def predict_disease(image_data):
    try:
        image = Image.open(image_data).convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return predicted_class, confidence
    except Exception as e:
        return "Error", str(e)

# ---------------------------
# Gemma API Query
# ---------------------------
def query_gemma(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": "gemma3n:e2b",
                "prompt": prompt,
                "stream": False
            }
        )
        if response.status_code == 200:
            output = response.json().get("response", "").strip()
            return output if output else "Gemma returned no response."
        else:
            return f"Failed to contact Gemma (status code {response.status_code})."
    except Exception as e:
        return f"Error contacting Gemma: {str(e)}"

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="AI Symptom Checker", page_icon="🍃", layout="wide")

# ---------------------------
# CSS Styling
# ---------------------------
st.markdown("""
    <style>
    html, body, .main {
        background-color: #000000;
        color: white;
    }
    .stTextInput > div > input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px;
        padding: 0.5rem;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stDownloadButton > button {
        background-color: #444;
        color: white;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 8px;
        color: #ffffff;
        font-weight: bold;
        padding: 1rem 2rem;
        font-size: 18px;
        margin: 0 auto;
    }
    .stTabs [aria-selected="true"] {
        background-color: #228B22;
        color: white;
    }
    .stTabs {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header + Centered Logo
# ---------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_path = "logo.png"
if Path(logo_path).exists():
    logo_base64 = get_base64_image(logo_path)
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{logo_base64}' width='180'><br>
            <h1>🍃🩺 CureGenie</h1>
            <h4 style='color: gray;'>Powered by TensorFlow & Gemma via Ollama (Fully Offline)</h4>
        </div><hr>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("⚠️ 'logo.png' not found. Please make sure it's in the same folder as app.py.")

# ---------------------------
# Tabs for Plant / Human
# ---------------------------
tab1, tab2 = st.tabs(["🌿 Plant Diagnosis", "👨‍⚕ Human Diagnosis"])

# ---------------------------
# PLANT TAB
# ---------------------------
with tab1:
    st.markdown("### 📷 Upload a plant leaf image (optional):")
    image_file = st.file_uploader("Choose a plant image", type=["jpg", "jpeg", "png"])

    st.markdown("### 🗘 Describe visible plant symptoms:")
    plant_description = st.text_area("e.g., yellowing leaves, black spots...")

    if st.button("🔍 Analyze Plant Symptoms"):
        prompt_parts = []

        if image_file:
            st.image(image_file, caption="🗉 Uploaded Plant Image", use_column_width=True)
            predicted_class, confidence = predict_disease(image_file)
            if predicted_class == "Error":
                st.error(f"❌ Image Prediction Error: {confidence}")
            else:
                st.success(f"✅ Image Prediction: {predicted_class} ({confidence}% confidence)")
                prompt_parts.append(f"The uploaded image shows: {predicted_class} ({confidence}% confidence).")

        if plant_description.strip():
            prompt_parts.append(f"User's description: \"{plant_description.strip()}\"")

        if not prompt_parts:
            st.warning("⚠ Please upload an image or enter symptoms.")
        else:
            final_prompt = (
                "You are a plant disease diagnosis expert. Based on the image and description below, "
                "identify the disease, give treatment and prevention steps:\n\n" + "\n".join(prompt_parts)
            )
            with st.spinner("🤖 Gemma is analyzing..."):
                gemma_response = query_gemma(final_prompt)
            st.markdown("### 🧠 Gemma's Diagnosis & Advice:")
            st.write(gemma_response)
            st.download_button("🗕 Download Result", gemma_response, file_name="plant_diagnosis.txt")

# ---------------------------
# HUMAN TAB
# ---------------------------
with tab2:
    st.markdown("### 🗘 Describe human symptoms:")
    human_input = st.text_area("e.g., fever, cough, stomach pain...")

    if st.button("🔬 Analyze Human Symptoms"):
        if human_input.strip() == "":
            st.warning("⚠ Please describe the symptoms.")
        else:
            prompt = (
                "You are a medical assistant AI. A user describes their symptoms:\n\n"
                f"\"{human_input.strip()}\"\n\n"
                "Give the likely condition, medications, and prevention (simple explanation)."
            )
            with st.spinner("🤖 Gemma is analyzing..."):
                gemma_response = query_gemma(prompt)
            st.markdown("### 🧠 Gemma's Diagnosis & Advice:")
            st.write(gemma_response)
            st.download_button("🗕 Download Result", gemma_response, file_name="human_diagnosis.txt")