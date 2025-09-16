from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np
import os
import requests

app = Flask(__name__)

# ----------------------------
# MODEL SECTION (Modified)
# ----------------------------

# URL of your YOLOv8 model from GitHub Release
model_url = "https://github.com/username/repo/releases/download/v1.0/best.pt"

# Local path to save the model
model_path = "best.pt"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading YOLO model...")
    r = requests.get(model_url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")

# Load the YOLOv8 model
model = YOLO(model_path)

# ----------------------------
# VIDEO SOURCE
# ----------------------------

# Use webcam (0) or replace with video file path
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame, conf=0.5, verbose=False)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------------------
# FLASK ROUTES
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------------
# RUN APP
# ----------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)