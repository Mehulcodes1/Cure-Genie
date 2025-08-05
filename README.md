🌿🧍 Offline AI Symptom Checker

An offline AI-powered tool that diagnoses **plant diseases from leaf images and text descriptions**, and also provides **human symptom analysis** — powered by:

- 🧠 [Gemma 3n:e2b](https://ollama.com/library/gemma) via Ollama (offline LLM)
- 🌱 TensorFlow (.h5 model) for plant disease detection
- ⚡ Streamlit for interactive UI

---

## 📦 Features

- 🔍 **Plant Disease Detection** using trained `.h5` model and symptom text
- 💬 **Human Health Analysis** via local Gemma LLM
- 📷 Upload plant images (JPG/PNG)
- 📝 Enter natural symptom descriptions
- 🛜 Fully **offline and privacy-preserving**
- 💾 Download diagnosis as text file

---

## 📁 Folder Structure

SYMOTOM_CHECKER/
├── dataset/ # (Optional) Your training data
├── plantenv/ # Virtual environment (ignored in Git)
├── app.py # ✅ Main Streamlit app
├── plant_disease_model.h5 # ✅ Trained TensorFlow model
├── train_model.py # (Optional) Model training script
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🛠️ How to Run

### 1. 🔧 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/)
- Gemma model pulled locally:
  ```bash
  ollama pull gemma3n:e2b
2. 📦 Create Virtual Environment
bash
Copy
Edit
python -m venv plantenv
plantenv\Scripts\activate    # Windows
source plantenv/bin/activate # macOS/Linux
3. 🧪 Install Requirements
bash
Copy
Edit
pip install streamlit tensorflow pillow requests
4. 🚀 Run the App
bash
Copy
Edit
streamlit run app.py
Then open: http://localhost:8501

💡 How It Works
For plants:

Upload a leaf image → Classifies disease using .h5 model

OR enter text → Sends to Gemma with image prediction (if any)

For humans:

Enter symptoms → Gemma gives diagnosis, medication, prevention

✅ Notes
No internet needed once Gemma is downloaded.

To retrain the image model, use train_model.py.

🤖 Made by a Beginner | For the Gemma 3n Hackathon
Built as part of a personal learning project and hackathon submission by Mehul Srivastava.