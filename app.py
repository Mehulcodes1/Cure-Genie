import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
st.set_page_config(page_title="AI Symptom Checker", page_icon="🍃", layout="wide")

import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import base64

# ---------------------------
# Load TensorFlow Model
# ---------------------------
try:
    model = tf.keras.models.load_model("plant_disease_model.h5")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.info("Make sure 'plant_disease_model.h5' is in the same folder as this script")
    model = None

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
# Gemma Query
# ---------------------------
def query_gemma(prompt):
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code != 200:
            return "❌ Ollama is not running. Please start it with 'ollama serve'."
        
        models_data = test_response.json()
        available_models = [model["name"] for model in models_data.get("models", [])]
        model_to_use = None
        possible_models = ["gemma2b", "gemma:2b", "gemma2:7b", "gemma:7b", "llama3.2:3b", "llama2:7b"]
        
        for model_name in possible_models:
            if any(model_name in available for available in available_models):
                model_to_use = model_name
                break
        
        if not model_to_use:
            return "❌ No suitable model found. Try pulling with: ollama pull gemma3n:e2b"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 400
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            output = result.get("response", "").strip()
            return output if output else "Gemma returned no response."
        else:
            return f"Gemma error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Gemma request timed out."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Run: ollama serve"
    except Exception as e:
        return f"Gemma error: {str(e)}"

# ---------------------------
# Style + Logo
# ---------------------------
st.markdown("""<style>
    html, body, .main { background-color: #000; color: white; }
    .stTextInput > div > input, .stTextArea textarea {
        background-color: #fff !important; color: #000 !important;
        border-radius: 10px; padding: 0.5rem; font-size: 16px;
    }
    .stButton > button, .stDownloadButton > button {
        background-color: #4CAF50; color: white;
        border-radius: 8px; padding: 0.5rem 1rem; font-weight: bold;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e; border-radius: 8px; color: #fff;
        font-weight: bold; padding: 1rem 2rem; font-size: 18px;
    }
    .stTabs [aria-selected="true"] { background-color: #228B22; }
</style>""", unsafe_allow_html=True)

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

logo_path = "logo.png"
logo_base64 = get_base64_image(logo_path)

if logo_base64:
    st.markdown(f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{logo_base64}' width='180'><br>
            <h1>🍃🩺 CureGenie</h1>
            <h4 style='color: gray;'>Powered by TensorFlow & Gemma via Ollama (Fully Offline)</h4>
        </div><hr>""", unsafe_allow_html=True)
else:
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🍃🩺 CureGenie</h1>
            <h4 style='color: gray;'>Powered by TensorFlow & Gemma via Ollama (Fully Offline)</h4>
        </div><hr>""", unsafe_allow_html=True)

# ---------------------------
# Status Check
# ---------------------------
try:
    ollama_check = requests.get("http://localhost:11434/api/tags", timeout=3)
    if ollama_check.status_code == 200:
        models_info = ollama_check.json()
        model_count = len(models_info.get("models", []))
        st.info(f"✅ Ollama running with {model_count} model(s)")
    else:
        st.warning("⚠️ Ollama is not responding correctly.")
except:
    st.warning("⚠️ Ollama not running or not installed.")

# ---------------------------
# Tabs for Diagnosis
# ---------------------------
tab1, tab2 = st.tabs(["🌿 Plant Diagnosis", "👨‍⚕ Human Diagnosis"])

# 🌿 Plant Tab
with tab1:
    st.markdown("### 📷 Upload a plant leaf image (optional):")
    image_file = st.file_uploader("Choose a plant image", type=["jpg", "jpeg", "png"])

    st.markdown("### 🗘 Describe visible plant symptoms:")
    plant_description = st.text_area("e.g., yellowing leaves, black spots...")

    if st.button("🔍 Analyze Plant Symptoms"):
        prompt_parts = []

        if image_file:
            st.image(image_file, caption="🗉 Uploaded Plant Image", use_column_width=True)
            if model:
                predicted_class, confidence = predict_disease(image_file)
                if predicted_class == "Error":
                    st.error(f"❌ Image Prediction Error: {confidence}")
                else:
                    st.success(f"✅ Image Prediction: {predicted_class} ({confidence}% confidence)")
                    prompt_parts.append(f"The uploaded image shows: {predicted_class} ({confidence}% confidence).")
            else:
                st.warning("⚠ Model not loaded.")

        if plant_description.strip():
            prompt_parts.append(f"User's description: \"{plant_description.strip()}\"")

        if not prompt_parts:
            st.warning("⚠ Please upload an image or enter symptoms.")
        else:
            final_prompt = (
                "You are a plant disease expert AI. Analyze the following image and/or symptoms, "
                "identify the most likely disease, and provide detailed treatment and prevention steps. "
                "Also suggest medications or remedies that can be found at home or locally nearby:\n\n" +
                "\n".join(prompt_parts)
            )
            with st.spinner("🤖 Gemma is analyzing..."):
                gemma_response = query_gemma(final_prompt)
            st.markdown("### 🧠 Gemma's Diagnosis & Advice:")
            st.write(gemma_response)
            st.download_button("🗕 Download Result", gemma_response, file_name="plant_diagnosis.txt")

# 👨‍⚕ Human Tab
with tab2:
    st.markdown("### 🗘 Describe human symptoms:")
    human_input = st.text_area("e.g., fever, cough, stomach pain...")

    if st.button("🔬 Analyze Human Symptoms"):
        if human_input.strip() == "":
            st.warning("⚠ Please describe the symptoms.")
        else:
            prompt = (
                "You are a helpful medical assistant. A user describes the following symptoms:\n\n"
                f"\"{human_input.strip()}\"\n\n"
                "Based on this, suggest the most likely illness. Also provide recommended medication and home remedies "
                "that are accessible and simple to follow. Add prevention steps too."
            )
            with st.spinner("🤖 Gemma is analyzing..."):
                gemma_response = query_gemma(prompt)
            st.markdown("### 🧠 Gemma's Diagnosis & Advice:")
            st.write(gemma_response)
            st.download_button("🗕 Download Result", gemma_response, file_name="human_diagnosis.txt")

# 🛠 Setup Help
with st.expander("🛠 Quick Setup Help (Click if Gemma not working)"):
    st.markdown("""
    **Steps to fix if Gemma isn’t working:**
    
    1. **Install Ollama** → https://ollama.ai  
    2. **Pull a model:**  
    ```bash
    ollama pull gemma3n:e2b
    ```
    3. **Start Ollama:**
    ```bash
    ollama serve
    ```
    4. **Test it works:**
    ```bash
    ollama list
    ```
    """)
    if st.button("🧪 Test Gemma Connection"):
        test_result = query_gemma("Just reply 'Working' if you're online.")
        if "working" in test_result.lower():
            st.success("✅ Gemma is working fine!")
        else:
            st.error("❌ Still broken. Here's the response:")
            st.write(test_result)
