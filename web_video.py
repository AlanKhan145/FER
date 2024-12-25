import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from prepocessing import *

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Global variable to hold the video capture
video_capture = None

def generate_video():
    """Video frame generator."""
    while True:
        if video_capture:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Process the frame
            # Lấy kích thước ban đầu của ảnh
            h, w = frame.shape[:2]

            # Chỉnh kích thước mới, giữ tỷ lệ
            new_width = 300
            new_height = int((new_width / w) * h)

            # Resize ảnh
            frame = cv2.resize(frame, (new_width, new_height))

            processed_frame = process_frame(frame)
            _, jpeg = cv2.imencode('.jpg', processed_frame)
            frame_data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
        else:
            # If video is not playing, just keep sending blank frames to keep stream alive
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('video.html')  # Render the HTML page

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and start processing."""
    global video_capture
    file = request.files['video_file']
    filename = "uploads/video.mp4"
    file.save(filename)

    video_capture = cv2.VideoCapture(filename)
    if not video_capture.isOpened():
        return jsonify({"error": "Unable to open video file"}), 400
    return jsonify({"message": "Video uploaded successfully"}), 200

@app.route('/video_feed')
def video_feed():
    """Video stream route."""
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Stop the video playback."""
    global video_capture
    if video_capture and video_capture.isOpened():
        video_capture.release()
        video_capture = None
    return jsonify({"message": "Video stopped successfully"}), 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
