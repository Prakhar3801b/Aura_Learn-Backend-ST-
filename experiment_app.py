import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import google.generativeai as genai

# ===============================
# Gemini Configuration (FIXED)
# ===============================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Aura_Learn Experiment",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Experiment Data
# ===============================
EXPERIMENTS = {
    "physics": {
        "title": "Physics: Pendulum Experiment",
        "context": "You are an AI Lab Assistant for a Physics Pendulum experiment.",
        "steps": [
            {"text": "Calibration", "desc": "Aligning sensors with gravity vector"},
            {"text": "Displacement", "desc": "Potential energy increases"},
            {"text": "Release", "desc": "Kinetic energy increases"}
        ]
    },
    "chemistry": {
        "title": "Chemistry: Titration",
        "context": "You are an AI Lab Assistant for a Chemistry Titration experiment.",
        "steps": [
            {"text": "Reactants", "desc": "HCl + NaOH"},
            {"text": "Neutralization", "desc": "pH approaches 7"},
            {"text": "End Point", "desc": "Indicator changes color"}
        ]
    },
    "biology": {
        "title": "Biology: Plant Cell",
        "context": "You are an AI Lab Assistant for a Biology experiment.",
        "steps": [
            {"text": "Cell Wall", "desc": "Rigid outer layer"},
            {"text": "Nucleus", "desc": "Genetic control center"},
            {"text": "Chloroplast", "desc": "Photosynthesis"}
        ]
    }
}

# ===============================
# Get Experiment from URL
# ===============================
params = st.query_params
experiment_key = params.get("experiment", "physics")
data = EXPERIMENTS.get(experiment_key, EXPERIMENTS["physics"])

# ===============================
# Sidebar (AI BOT)
# ===============================
with st.sidebar:
    st.title("🧪 Experiment Control")
    st.subheader(data["title"])

    st.markdown("---")
    st.subheader("🤖 AURA-Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask AURA about this experiment"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not GEMINI_AVAILABLE:
                response = "⚠️ Gemini API key not configured."
            else:
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(
                    f"{data['context']}\nUser: {prompt}\nAnswer briefly."
                ).text

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ===============================
# MAIN AREA — CLOUD CAMERA (FIXED)
# ===============================
st.subheader("📷 AR Sight (Browser Camera)")

uploaded = st.camera_input("Enable camera")

if uploaded:
    image = Image.open(uploaded)
    frame = np.array(image)

    h, w, _ = frame.shape
    roi_size = 200
    x1, y1 = w // 2 - roi_size // 2, h // 2 - roi_size // 2
    x2, y2 = x1 + roi_size, y1 + roi_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "FOCUS AREA", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, use_container_width=True)

    step = data["steps"][int(time.time()) % len(data["steps"])]
    st.info(f"**Current Step:** {step['text']} — {step['desc']}")
else:
    st.warning("Enable camera to start AR view.")
