import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from prepocessing import preprocess_frame

# Load mô hình phân loại cảm xúc
emotion_classifier = load_model('models/vgg.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Thông số giao diện
window_name = "PREDICT EMOTION BASED ON HUMAN FACES"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
frame = np.ones((600, 800, 3), dtype=np.uint8) * 255

is_running = False
button_specs = {
    "START": {"pos": (50, 520), "color": (0, 200, 0), "action": "start"},
    "STOP": {"pos": (300, 520), "color": (0, 0, 200), "action": "stop"},
    "EXIT": {"pos": (550, 520), "color": (200, 0, 0), "action": "exit"}
}

def draw_interface(frame, is_running):
    """Vẽ giao diện và trạng thái."""
    cv2.rectangle(frame, (0, 0), (800, 100), (200, 200, 200), -1)
    cv2.putText(frame, "PREDICT EMOTION BASED ON HUMAN FACES", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    for button, specs in button_specs.items():
        x, y = specs["pos"]
        color = specs["color"]
        cv2.rectangle(frame, (x, y), (x + 200, y + 50), color, -1)
        cv2.putText(frame, button, (x + 60, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Hiển thị trạng thái
    status_text = "RUNNING" if is_running else "STOPPED"
    status_color = (0, 255, 0) if is_running else (0, 0, 255)
    cv2.putText(frame, f"STATUS: {status_text}", (50, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

def process_frame(video_frame):
    """Xử lý dự đoán cảm xúc trên một frame video."""
    processed_faces = preprocess_frame(video_frame)
    for face, face_coords in processed_faces:
        try:
            predictions = emotion_classifier.predict(np.expand_dims(face, axis=0), verbose=0)
            emotion_index = np.argmax(predictions)
            emotion_text = f"{emotion_labels[emotion_index]} ({np.max(predictions)*100:.2f}%)"
            x, y, w, h = face_coords
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(video_frame, emotion_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Lỗi dự đoán: {e}")
    return video_frame

def mouse_callback(event, x, y, flags, param):
    """Xử lý sự kiện chuột."""
    global is_running, running
    if event == cv2.EVENT_LBUTTONDOWN:
        for button, specs in button_specs.items():
            bx, by = specs["pos"]
            if bx <= x <= bx + 200 and by <= y <= by + 50:
                action = specs["action"]
                if action == "start":
                    is_running = True
                elif action == "stop":
                    is_running = False
                elif action == "exit":
                    running = False

cv2.setMouseCallback(window_name, mouse_callback)

# Sử dụng webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Không thể truy cập camera.")
    exit()

# Vòng lặp chính
running = True
try:
    while running:
        frame[:] = 255
        if is_running:
            ret, video_frame = video_capture.read()
            if ret:
                video_frame = process_frame(video_frame)
                video_frame = cv2.resize(video_frame, (300, 300))
                frame[100:400, 170:470] = video_frame
        draw_interface(frame, is_running)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    video_capture.release()
    cv2.destroyAllWindows()
