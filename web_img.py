import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from prepocessing import *

# Khởi tạo Flask
app = Flask(__name__)

# Thư mục lưu trữ ảnh tải lên và ảnh đã xử lý
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
PREDICT_IMG_FOLDER = 'uploads/predicted'  # Đảm bảo lưu trong thư mục uploads/predicted
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICT_IMG_FOLDER'] = PREDICT_IMG_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load mô hình phân loại cảm xúc
emotion_classifier = load_model('models/vgg.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('img.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Tạo thư mục 'uploads' nếu nó chưa tồn tại
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Xử lý ảnh và lưu ảnh đã xử lý
        processed_image_path = process_and_save_image(filepath, filename)

        # Trả về đường dẫn ảnh đã xử lý để hiển thị lại
        processed_image_url = f"/{PREDICT_IMG_FOLDER}/{processed_image_path.split(os.sep)[-1]}"

        return jsonify({"processed_image_url": processed_image_url})
    else:
        return jsonify({"error": "File type not allowed"}), 400

def process_and_save_image(image_path, original_filename):
    # Đọc ảnh và phát hiện khuôn mặt
    image = cv2.imread(image_path)
    processed_faces = preprocess_frame(image)

    # Xử lý ảnh và vẽ khung quanh khuôn mặt
    for face, face_coords in processed_faces:
        predictions = emotion_classifier.predict(np.expand_dims(face, axis=0), verbose=0)
        emotion_index = np.argmax(predictions)
        emotion_text = emotion_labels[emotion_index]
        x, y, w, h = face_coords
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, emotion_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Lưu ảnh đã xử lý vào thư mục 'uploads/predicted'
    processed_filename = f"processed_{original_filename}"
    processed_image_path = os.path.join(app.config['PREDICT_IMG_FOLDER'], processed_filename)

    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(app.config['PREDICT_IMG_FOLDER']):
        os.makedirs(app.config['PREDICT_IMG_FOLDER'])

    cv2.imwrite(processed_image_path, image)

    return processed_image_path

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/predicted/<filename>')
def send_predicted_file(filename):
    return send_from_directory(app.config['PREDICT_IMG_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
