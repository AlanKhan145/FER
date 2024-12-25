import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from prepocessing import preprocess_frame

# Load mô hình phân loại cảm xúc
emotion_classifier = load_model('models/vgg.keras')

# Nhãn cảm xúc
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Giao diện hiển thị
window_name = "PREDICT EMOTION BASED ON HUMAN FACES"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Cho phép thay đổi kích thước
frame = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Biến trạng thái
is_running = False

# Thông số nút giao diện
button_specs = {
    "START": {"pos": (50, 520), "color": (0, 200, 0), "action": "start"},
    "STOP": {"pos": (300, 520), "color": (0, 0, 200), "action": "stop"},
    "EXIT": {"pos": (550, 520), "color": (200, 0, 0), "action": "exit"}
}


# Vẽ giao diện
def draw_interface(frame):
    cv2.rectangle(frame, (0, 0), (800, 100), (200, 200, 200), -1)
    cv2.putText(frame, "PREDICT EMOTION BASED ON HUMAN FACES", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

    # Vẽ các nút
    for button, specs in button_specs.items():
        x, y = specs["pos"]
        color = specs["color"]
        cv2.rectangle(frame, (x, y), (x + 200, y + 50), color, -1)
        cv2.putText(frame, button, (x + 60, y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# Xử lý sự kiện chuột
def mouse_callback(event, x, y, flags, param):
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
                    running = False  # Thoát vòng lặp chính


cv2.setMouseCallback(window_name, mouse_callback)

# Sử dụng video từ file
video_path = 'video1.mp4'
video_capture = cv2.VideoCapture(video_path)

# Kiểm tra xem video có hoạt động không
if not video_capture.isOpened():
    print("Không thể mở video.")
    exit()

# Vòng lặp chính
running = True
while running:
    frame[:] = 255  # Làm mới frame

    if is_running:
        ret, video_frame = video_capture.read()
        if ret:
            # Tiền xử lý toàn bộ frame
            processed_faces = preprocess_frame(video_frame)

            for face, face_coords in processed_faces:
                # Dự đoán cảm xúc từ khuôn mặt đã xử lý
                predictions = emotion_classifier.predict(np.expand_dims(face, axis=0), verbose=0)
                emotion_index = np.argmax(predictions)

                emotion_text = f"{emotion_labels[emotion_index]} ({np.max(predictions) * 100:.2f}%)"

                # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị cảm xúc
                x, y, w, h = face_coords
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(video_frame, emotion_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Giữ nguyên tỷ lệ video gốc khi nhúng vào giao diện
            video_height, video_width = video_frame.shape[:2]
            scale = min(300 / video_width, 300 / video_height)
            resized_width = int(video_width * scale)
            resized_height = int(video_height * scale)
            resized_frame = cv2.resize(video_frame, (resized_width, resized_height))

            # Căn chỉnh vị trí để nhúng vào giao diện
            top_left_x = 170 + (300 - resized_width) // 2
            top_left_y = 100 + (300 - resized_height) // 2
            frame[top_left_y:top_left_y + resized_height, top_left_x:top_left_x + resized_width] = resized_frame

    draw_interface(frame)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
