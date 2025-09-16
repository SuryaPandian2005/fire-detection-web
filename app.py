from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load the YOLOv8 model
model_path = r"D:\Fire Detction project\model\best.pt"
model = YOLO(model_path)

# Your video source (webcam or file)
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Read a frame from the video source
        success, frame = cap.read()
        if not success:
            break

        # Run inference on the frame
        # The 'model' object handles all preprocessing and postprocessing
        results = model(frame, conf=0.5, verbose=False)

        # Get the annotated frame with bounding boxes and labels
        annotated_frame = results[0].plot()

        # Encode the annotated frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)