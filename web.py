from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from flask_socketio import SocketIO
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from prepocessing import *
import os
from web_camera import *
from web_video import *
from web_img import *
# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Set up folders and model
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
PREDICT_IMG_FOLDER = os.path.join(UPLOAD_FOLDER, 'predicted')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_IMG_FOLDER'] = PREDICT_IMG_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

emotion_classifier = load_model('models/vgg.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

video_capture = None

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_and_save_image(image_path, original_filename):
    img = cv2.imread(image_path)
    processed_faces = preprocess_frame(img)

    for face, face_coords in processed_faces:
        predictions = emotion_classifier.predict(np.expand_dims(face, axis=0), verbose=0)
        emotion_index = np.argmax(predictions)
        emotion_text = emotion_labels[emotion_index]
        x, y, w, h = face_coords
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    processed_filename = f"processed_{original_filename}"
    processed_image_path = os.path.join(app.config['PREDICT_IMG_FOLDER'], processed_filename)

    if not os.path.exists(app.config['PREDICT_IMG_FOLDER']):
        os.makedirs(app.config['PREDICT_IMG_FOLDER'])

    cv2.imwrite(processed_image_path, img)
    return processed_image_path

def generate_video():
    global video_capture
    while True:
        if video_capture and video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            h, w = frame.shape[:2]
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
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n\r\n')

# Routes
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/camera')
def camera():
    return render_template('web_camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    if video_capture and video_capture.isOpened():
        video_capture.release()
        video_capture = None
    return jsonify({"message": "Camera stopped successfully"}), 200

@app.route('/img')
def img():
    return render_template('web_img.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file.save(filepath)
        processed_image_path = process_and_save_image(filepath, filename)
        processed_image_url = f"/{PREDICT_IMG_FOLDER}/{os.path.basename(processed_image_path)}"
        return jsonify({"processed_image_url": processed_image_url})
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/predicted/<filename>')
def send_predicted_file(filename):
    return send_from_directory(app.config['PREDICT_IMG_FOLDER'], filename)

@app.route('/video')
def video():
    return render_template('web_video.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_capture
    file = request.files['video_file']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(filename)
    video_capture = cv2.VideoCapture(filename)
    if not video_capture.isOpened():
        return jsonify({"error": "Unable to open video file"}), 400
    return jsonify({"message": "Video uploaded successfully"}), 200

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_capture
    if video_capture and video_capture.isOpened():
        video_capture.release()
        video_capture = None
    return jsonify({"message": "Video stopped successfully"}), 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
