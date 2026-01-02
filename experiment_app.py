import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import google.generativeai as genai

# ===============================
# Gemini Configuration (SAFE)
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
# SAFE GEMINI RESPONSE FUNCTION
# ===============================
def safe_gemini_response(prompt_text):
    try:
        # Dynamically find a model that supports generateContent
        available_model = None
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                available_model = m.name
                break

        if not available_model:
            return "⚠️ No compatible AI model available for this API key."

        model = genai.GenerativeModel(available_model)

        result = model.generate_content(
            f"{data['context']}\nUser: {prompt_text}\nAnswer briefly."
        )

        if hasattr(result, "text") and result.text:
            return result.text.strip()
        else:
            return "⚠️ AI returned an empty response."

    except Exception as e:
        return f"⚠️ AI error: {str(e)}"

        if hasattr(result, "text") and result.text:
            return result.text.strip()
        else:
            return "⚠️ AI returned no content. Try rephrasing your question."

    except Exception as e:
        return f"⚠️ AI error: {str(e)}"


        if hasattr(result, "text") and result.text:
            return result.text.strip()
        else:
            return "⚠️ AI response was empty. Please try again."

    except Exception:
        return "⚠️ AI service is temporarily unavailable. Please try again later."

# ===============================
# Sidebar — AI BOT
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
                response = safe_gemini_response(prompt)

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ===============================
# MAIN AREA — AR CAMERA (FOCUS + ENTRY LOGIC)
# ===============================
st.subheader("📷 AR Sight (Focus Zone Enabled)")

uploaded = st.camera_input("Bring an object into the focus box")

# Store previous ROI for entry detection
if "prev_roi" not in st.session_state:
    st.session_state.prev_roi = None

if uploaded:
    image = Image.open(uploaded)
    frame = np.array(image)

    h, w, _ = frame.shape

    # ---- CENTER FOCUS BOX ----
    roi_size = int(min(h, w) * 0.45)
    x1 = w // 2 - roi_size // 2
    y1 = h // 2 - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    # ---- DARKEN OUTSIDE AREA ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # ---- DRAW FOCUS BOX ----
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 6)

    # ---- CENTER CROSSHAIR ----
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 0), 3)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 0), 3)

    cv2.putText(
        frame,
        "FOCUS AREA",
        (x1 + 15, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3
    )

    # ---- ROI PROCESSING ----
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    highlight = False

    if st.session_state.prev_roi is not None:
        diff = cv2.absdiff(st.session_state.prev_roi, gray_roi)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        change_ratio = np.sum(thresh) / thresh.size

        if change_ratio > 0.02:
            highlight = True

    st.session_state.prev_roi = gray_roi

    # ---- HIGHLIGHT ONLY NEW ENTRIES ----
    if highlight:
        edges = cv2.Canny(gray_roi, 60, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cnt[:, :, 0] += x1
            cnt[:, :, 1] += y1
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 4)

        st.success("✔ Object detected entering focus area")
    else:
        st.info("Waiting for object to enter focus area")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, use_container_width=True)

    step = data["steps"][int(time.time()) % len(data["steps"])]
    st.info(f"**Current Step:** {step['text']} — {step['desc']}")

else:
    st.warning("Enable camera and bring an object into the focus box.")



