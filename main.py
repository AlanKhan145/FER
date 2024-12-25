import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from prepocessing import *

# Load mô hình phân loại cảm xúc
emotion_classifier = load_model('models/vgg.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Giao diện hiển thị
window_name = "Emotion Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
frame = np.ones((700, 1000, 3), dtype=np.uint8) * 255

# Biến trạng thái
video_path = None
image_path = None
video_capture = None
is_video = False
is_running = False

# Thông số nút giao diện
button_specs = {
    "UPLOAD": {"pos": (100, 620), "color": (0, 123, 255), "action": "upload"},
    "START": {"pos": (300, 620), "color": (0, 200, 0), "action": "start"},
    "EXIT": {"pos": (700, 620), "color": (255, 0, 0), "action": "exit"}
}

# Vẽ giao diện
def draw_interface(frame):
    cv2.putText(frame, "Emotion Detection Using AI", (250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(frame, "Choose a video and start detecting emotions", (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    for button, specs in button_specs.items():
        x, y = specs["pos"]
        color = specs["color"]
        cv2.rectangle(frame, (x, y), (x + 150, y + 50), color, -1)
        cv2.putText(frame, button, (x + 30, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Hàm chọn video hoặc ảnh
def upload_video_or_image():
    global video_path, image_path, video_capture, is_video
    root = tk.Tk()
    root.withdraw()

    # Chọn video hoặc ảnh
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("Image Files", "*.jpg *.jpeg *.png *.bmp")])

    if file_path:
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):  # Kiểm tra xem có phải video không
            video_path = file_path
            video_capture = cv2.VideoCapture(video_path)
            is_video = True  # Đánh dấu đây là video
            print(f"Video selected: {video_path}")
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Kiểm tra xem có phải ảnh không
            image_path = file_path
            is_video = False  # Đánh dấu đây là ảnh
            print(f"Image selected: {image_path}")

# Sự kiện chuột
def mouse_callback(event, x, y, flags, param):
    global is_running, video_capture, running, button_specs
    if event == cv2.EVENT_LBUTTONDOWN:
        for button, specs in button_specs.items():
            bx, by = specs["pos"]
            if bx <= x <= bx + 150 and by <= y <= by + 50:
                action = specs["action"]
                if action == "upload":
                    upload_video_or_image()  # Gọi hàm chọn video hoặc ảnh
                elif action == "start":
                    is_running = True  # Đánh dấu trạng thái bắt đầu
                elif action == "exit":
                    global running
                    running = False  # Thoát chương trình

cv2.setMouseCallback(window_name, mouse_callback)

# Vòng lặp chính
running = True
is_video = False  # Mặc định là không phải video
while running:
    frame[:] = 255  # Làm trắng frame (hoặc có thể tùy chỉnh màu nền)

    if is_running:
        if is_video and video_capture:
            ret, video_frame = video_capture.read()
            if ret:
                processed_faces = preprocess_frame(video_frame)
                for face, face_coords in processed_faces:
                    predictions = emotion_classifier.predict(np.expand_dims(face, axis=0), verbose=0)
                    emotion_index = np.argmax(predictions)
                    emotion_text = emotion_labels[emotion_index]
                    x, y, w, h = face_coords
                    cv2.rectangle(video_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(video_frame, emotion_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                video_frame = cv2.resize(video_frame, (400, 300))
                frame[150:450, 300:700] = video_frame
            else:
                is_running = False  # Khi video hết, dừng chương trình
        elif not is_video and image_path:
            image = cv2.imread(image_path)
            processed_faces = preprocess_frame(image)
            for face, face_coords in processed_faces:
                predictions = emotion_classifier.predict(np.expand_dims(face, axis=0), verbose=0)
                emotion_index = np.argmax(predictions)
                emotion_text = emotion_labels[emotion_index]
                x, y, w, h = face_coords
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(image, emotion_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            image_resized = cv2.resize(image, (400, 300))
            frame[150:450, 300:700] = image_resized

    draw_interface(frame)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if video_capture:
    video_capture.release()
cv2.destroyAllWindows()
