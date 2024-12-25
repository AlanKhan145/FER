from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename
from prepocessing import *
import os

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Set up webcam video capture
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Unable to access camera.")
    exit()

def generate_video():
    """Video frame generator."""
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)
        _, jpeg = cv2.imencode('.jpg', processed_frame)
        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    """Video stream route."""
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        return jsonify({"error": "Unable to access camera"}), 400
    return jsonify({"message": "Camera started successfully"}), 200

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global video_capture
    if video_capture.isOpened():
        video_capture.release()
    return "Camera stopped", 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
