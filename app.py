from flask import Flask, Response, render_template, jsonify
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load the YOLOv8 model
model_path = r"D:\Fire Detction project\model\best.pt"
model = YOLO(model_path)

# Video source (webcam or file)
cap = cv2.VideoCapture(0)

# Global variables to track fire detection
fire_detected = False
fire_lat = None
fire_lon = None

def generate_frames():
    global fire_detected, fire_lat, fire_lon
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()  # Original annotated frame

        # Check detected objects for fire or smoke
        detected_labels = results[0].names  # All class names
        detected_objects = results[0].boxes.cls if len(results[0].boxes) > 0 else []

        # Reset detection
        fire_detected = False
        fire_lat = None
        fire_lon = None

        # Overlay a custom message if fire or smoke is detected
        message = ""
        for cls_id in detected_objects:
            label = detected_labels[int(cls_id)]
            if label.lower() in ["fire", "smoke"]:
                message = f"⚠ ALERT: {label.upper()} DETECTED!"
                fire_detected = True

                # Optional: you can set approximate coordinates here
                # For demo purposes, random/static coordinates are used
                fire_lat = 10.9608   # Example: latitude
                fire_lon = 78.0784   # Example: longitude
                break  # Show first detected alert

        if message:
            # Put message on the frame
            cv2.putText(frame, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/dash.html')
def dash():
    return render_template('dash.html')

@app.route('/log.html')
def log():
    return render_template('log.html')

@app.route('/set.html')
def set():
    return render_template('set.html')

@app.route('/help.html')
def help_page():
    return render_template('help.html')



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# New endpoint to provide fire status and location
@app.route('/status')
def status():
    return jsonify({
        "fire": fire_detected,
        "lat": fire_lat,
        "lon": fire_lon
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
