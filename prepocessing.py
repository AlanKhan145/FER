import cv2
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN

# Hàm tiền xử lý frame để phát hiện khuôn mặt và dự đoán cảm xúc
def preprocess_frame(frame, target_size=(48, 48)):
    """
    Tiền xử lý ảnh (frame) để phát hiện và cắt khuôn mặt.
    :param frame: Frame ảnh từ camera.
    :param target_size: Kích thước mục tiêu của khuôn mặt (chiều cao, chiều rộng).
    :return: Danh sách các khuôn mặt đã xử lý.
    """
    # Khởi tạo MTCNN detector
    detector = MTCNN()

    # Phát hiện khuôn mặt
    results = detector.detect_faces(frame)

    # Danh sách các khuôn mặt đã xử lý
    processed_faces = []

    for result in results:
        x, y, w, h = result['box']
        face_image = frame[y:y + h, x:x + w]  # Cắt khuôn mặt từ ảnh gốc
        resized_image = cv2.resize(face_image, target_size)  # Thay đổi kích thước
        resized_image = resized_image.astype('float32') / 255.0  # Chuẩn hóa pixel
        processed_faces.append((resized_image, (x, y, w, h)))

    return processed_faces


# Hàm xử lý frame video để dự đoán cảm xúc
def process_frame(video_frame):
    emotion_classifier = load_model('models/vgg.keras')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    """Process the frame for emotion prediction."""
    processed_faces = preprocess_frame(video_frame)

    for face, face_coords in processed_faces:
        try:
            predictions = emotion_classifier.predict(np.expand_dims(face, axis=0), verbose=0)
            emotion_index = np.argmax(predictions)
            emotion_text = f"{emotion_labels[emotion_index]} ({np.max(predictions) * 100:.2f}%)"
            x, y, w, h = face_coords
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(video_frame, emotion_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in prediction: {e}")
    return video_frame

