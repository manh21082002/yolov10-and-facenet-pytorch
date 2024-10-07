import argparse
import pickle
import cv2
import time
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np

# Cài đặt đường dẫn cho YOLO và dữ liệu đã huấn luyện
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
args = vars(ap.parse_args())

# Load YOLO model
print("[INFO] loading YOLO model...")
yolo_model = YOLO("D:/AI_programing_lastterm/weight/best.pt")

# Load FaceNet (InceptionResnetV1)
print("[INFO] loading FaceNet model...")
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# Load known face encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Preprocessing cho FaceNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Mở webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Bắt đầu tính thời gian xử lý khung hình
    start_time = time.time()

    # Đọc khung hình từ webcam
    ret, frame = video_capture.read()

    # Sử dụng YOLO để phát hiện khuôn mặt
    results = yolo_model(frame)

    # Khởi tạo danh sách các tên khuôn mặt được phát hiện
    names = []

    # Duyệt qua các khuôn mặt được phát hiện
    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0].cpu().numpy())
        
        # Cắt ảnh khuôn mặt ra khỏi khung hình
        face_image = frame[y_min:y_max, x_min:x_max]

        # Chuyển ảnh sang dạng tensor và tính embedding bằng FaceNet
        face_tensor = preprocess(face_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            embedding = facenet_model(face_tensor).cpu().numpy()

        # So sánh với các khuôn mặt đã biết
        matches = []
        for known_embedding in data['encodings']:
            distance = np.linalg.norm(embedding - known_embedding)
            matches.append(distance)

        # Tìm khoảng cách ngắn nhất và xác định tên
        min_distance = min(matches)
        name = "Unknown"
        if min_distance < 0.7:  # Ngưỡng khoảng cách
            matchedIdx = matches.index(min_distance)
            name = data['names'][matchedIdx]

            # Tính độ chính xác: 1 - (distance / threshold)
            confidence = 1 - (min_distance / 0.7)  
        # Lưu lại tên khuôn mặt
        names.append(name)

        # Vẽ bounding box và tên lên khung hình
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        y = y_min - 15 if y_min - 15 > 15 else y_min + 15
        cv2.putText(frame, f"{name}: {confidence*100:.2f}%", (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Tính toán FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = 1 / total_time
    print(f"[INFO] FPS: {fps:.2f}")

    # Hiển thị khung hình
    cv2.imshow("Video", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
video_capture.release()
cv2.destroyAllWindows()
