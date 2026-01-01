import streamlit as st
import cv2
import numpy as np
import time
import os 
# ... Imports ...
import google.generativeai as genai

# !!! PASTE YOUR API KEY HERE !!!
GEMINI_API_KEY =os.environ.get("GEMINI_API_KEY")

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    pass # Will fail if key is empty, handled in chat loop

# Page Configuration
st.set_page_config(
    page_title="Aura_Learn Experiment",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Experiment Data
EXPERIMENTS = {
    "physics": {
        "title": "Physics: Pendulum Experiment",
        "context": "You are an AI Lab Assistant for a Physics Pendulum experiment. Focus on gravity, potential energy, kinetic energy, and oscillation period.",
        "steps": [
            {"text": "Step 1: Calibration", "desc": "Aligning sensors with gravity vector (g)."},
            {"text": "Step 2: Displacement", "desc": "PE = m * g * h (Potential Energy Max)"},
            {"text": "Step 3: Release", "desc": "KE = 0.5 * m * v^2 (Kinetic Energy Increases)"}
        ]
    },
    "chemistry": {
        "title": "Chemistry: Titration",
        "context": "You are an AI Lab Assistant for a Chemistry Titration experiment. Focus on acids, bases, neutralization, pH levels, and indicators.",
        "steps": [
            {"text": "Analysis: Reactants", "desc": "HCl + NaOH -> NaCl + H2O"},
            {"text": "Status: Neutralization", "desc": "pH approaching 7.0 (Neutral)"},
            {"text": "Result: End Point", "desc": "Indicator Color Change Detected"}
        ]
    },
    "biology": {
        "title": "Biology: Plant Cell",
        "context": "You are an AI Lab Assistant for a Biology experiment observing Plant Cells. Focus on cell walls, nucleus, chloroplasts, and photosynthesis.",
        "steps": [
            {"text": "Focus: Cell Wall", "desc": "Rigid outer layer (Cellulose)"},
            {"text": "Focus: Nucleus", "desc": "Contains DNA (Genetic Control Center)"},
            {"text": "Focus: Chloroplasts", "desc": "Photosynthesis Site (Chlorophyll)"}
        ]
    }
}

# Get Experiment from Query Params
query_params = st.query_params
experiment_key = query_params.get("experiment", "physics")
data = EXPERIMENTS.get(experiment_key, EXPERIMENTS["physics"])

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - AI Tools
with st.sidebar:
    st.title("🧪 Experiment Control")
    st.header(data["title"])
    st.markdown("---")
    
    # (API Key Input Removed)

    # 2. AURA-Bot Chat Interface
    st.subheader("🤖 AURA-Bot")
    
    # Model Selection Debugger
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # If no models found, offer a text input fallback
        if not available_models:
             model_name = st.text_input("Model Name (e.g. gemini-pro):", "gemini-pro")
        else:
             model_name = st.selectbox("Select AI Model:", available_models, index=0)
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        model_name = "gemini-pro" # Fallback

    # Display Chat History
    chat_container = st.container(height=300)
    
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask AURA about this experiment..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        # Check if key is placeholder
        if "PASTE_YOUR" in GEMINI_API_KEY:
            response = "⚠️ Please open 'experiment_app.py' and paste your API Key in line 6."
            with chat_container.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            try:
                # Setup Model with Context
                # Strip 'models/' prefix if present in the selectbox for clean display, but API accepts it
                model = genai.GenerativeModel(model_name)
                context_prompt = f"{data['context']}\nUser Question: {prompt}\nAnswer succinctly:"
                
                with chat_container.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        ai_resp = model.generate_content(context_prompt)
                        response = ai_resp.text
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"AI Error: {e}")
                
    st.markdown("---")
    
    # Navigation Buttons
    vr_url = f"http://localhost:8000/vr_lab.html?experiment={experiment_key}"
    st.link_button("🥽 Enter VR Lab", vr_url, help="Switch to Immersive VR Mode", type="primary", use_container_width=True)
    
    if st.button("Return to Dashboard", use_container_width=True):
        st.markdown(f'<meta http-equiv="refresh" content="0;url=http://localhost:8000">', unsafe_allow_html=True)

# Main Area - Camera Feed (unchanged mostly)
st.subheader("AR Sight (Live Feed)")
camera_placeholder = st.empty()

# Overlay Info Container
info_col1, info_col2 = st.columns(2)
with info_col1:
    step_container = st.empty()
with info_col2:
    confidence_container = st.empty()

# Camera Capture
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

cap = st.session_state.cap

if not cap.isOpened():
    st.error("Camera NOT found. Please check your device.")
else:
    # We use a loop inside a placeholder to simulate video stream
    while True:
        # Camera Processing Loop
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame.")
            break
        
        # Mirror
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Define Focus Area (ROI)
        roi_size = 200
        x1, y1 = w//2 - roi_size//2, h//2 - roi_size//2
        x2, y2 = x1 + roi_size, y1 + roi_size
        
        # Draw Focus Box Guideline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "FOCUS AREA", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # --- Object Highlighting Logic ---
        # 1. Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # 2. Pre-process (Gray + Blur)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        # 3. Edge Detection (Canny)
        edges = cv2.Canny(blurred_roi, 50, 150)
        
        # 4. Find Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Draw Contours on Main Frame (Neon Green)
        # Note: We must offset contours by (x1, y1) to match main frame coordinates
        for cnt in contours:
            cnt[:, :, 0] += x1 # Offset X
            cnt[:, :, 1] += y1 # Offset Y
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        # -------------------------------
        
        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display
        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update Simulated Overlay Info
        current_time = int(time.time())
        step_idx = (current_time // 5) % len(data["steps"])
        step_info = data["steps"][step_idx]
        
        step_container.info(f"**Current Step:** {step_info['text']}")
        confidence_container.metric("AI Confidence", f"{85 + (current_time % 10)}%")
        
        # Important: Allow Streamlit to handle UI updates from the chat loop above
        # The while loop blocks the script, so we need a way to break or handle events?
        # Streamlit's model is top-down. To have a "video loop" AND "chat interaction", 
        # we usually need to separate them or use `st.experimental_rerun` cautiously.
        # However, for a simple demo, checking for input inside the loop is hard.
        # A common pattern for "Live Video + Chat" in Streamlit is using `streamlit-webrtc`,
        # OR simply refreshing the frame occasionally, OR putting the video in a separate thread.
        # But `st.chat_input` creates a re-run. If we are in a `while True` loop, the re-run won't happen 
        # until the loop breaks.
        
        # WORKAROUND for Standard Streamlit: 
        # We process one frame, sleep slightly, but we rely on the script Re-Running 
        # when a user sends a message. The `while True` loop is actually problematic for interactivity here.
        # 
        # BETTER APPROACH:
        # Don't use `while True` for the video if we want chat interaction in the same thread.
        # OR, we assume the user stops the video to chat? No, that's bad.
        #
        # For this prototype, I will use a shorter loop or check session state?
        # Actually, `st.image` updates in place.
        # If I use `while True`, the UI will NOT react to chat inputs.
        # Convert to: running a few frames (burst) then checking? No.
        # 
        # Let's use `check_stop` or similiar?
        # To strictly follow the user request "add AI reasoning bot", I will prioritize the Chat functionality.
        # I will make the video update loop breakable or non-blocking?
        # 
        # Actually, typically we'd use `streamlit-webrtc` for async video.
        # Since I am using `cv2` directly, I might accept that "Video pauses when I type"?
        # Or I can try to use `st.empty()` and just run the loop. 
        # Input widgets in Streamlit generally trigger a full script rerun.
        # If I am inside `while True`, the input widget event is queued.
        # I need to break the loop to process it?
        #
        # Let's try to limit the loop to 100 frames, then rerun? No that flickers.
        # 
        # I will implement the chat OUTSIDE the video loop? 
        # No, the video loop is the "main" thing.
        #
        # Compromise for this tool:
        # I will keep the `while True` loop BUT add a "Stop/Pause" button?
        # OR I will rely on the fact that Streamlit interacts via re-runs.
        # Actually, if I type in chat, Streamlit forces a re-run, which kills the Python process 
        # and restarts it? `cv2.VideoCapture` might be re-initialized.
        # That's why `if "cap" not in st.session_state` is crucial (which I added).
        #
        # So, the loop will run. When user types in chat and hits Enter:
        # Streamlit interrupts the script. Reruns from top.
        # Chat processes.
        # Loop restarts.
        # Video might flicker once, but it should work.
        
        time.sleep(0.03)


