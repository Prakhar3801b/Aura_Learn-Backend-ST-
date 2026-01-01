from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import random

app = Flask(__name__, static_folder='.', template_folder='.')
app.config['SECRET_KEY'] = 'aura_secret'
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('process_frame')
def handle_frame(data):
    # data is expected to be a dict: {'image': 'base64...', 'domain': 'physics'}
    try:
        image_data = data.get('image', '')
        domain = data.get('domain', 'physics')
        
        if not image_data:
            return

        # Decode base64 
        # Remove header "data:image/jpeg;base64,"
        if ',' in image_data:
            header, encoded = image_data.split(',', 1)
        else:
            encoded = image_data

        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # --- AI LOGIC (MOCK) ---
        # Here we pretend to analyze the frame with OpenCV
        
        # 1. Calculate average brightness as a "metric"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # 2. Mock "Detection"
        confidence = min(99, int(brightness / 2.55 + random.randint(-5, 5)))
        
        # 3. Domain Logic
        result = {
            "confidence": f"{confidence}%",
            "status": "Analyzing...",
            "action_required": False
        }

        if confidence > 60:
            result["status"] = "Object Detected"
        else:
            result["status"] = "Searching..."
            result["action_required"] = True
            
        # Emit result back to client
        emit('ai_response', result)

    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    print("Starting Aura_Learn SocketIO Server...")
    print("Open http://localhost:5000")
    # socketio.run handles the server start
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
