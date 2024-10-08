import os
import pickle
import cv2
import time
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
from imutils import paths

# Load YOLO model
print("[INFO] loading YOLO model...")
yolo_model = YOLO("./weight/best.pt")

# Load FaceNet (InceptionResnetV1)
print("[INFO] loading FaceNet model...")
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# Load known face encodings
print("[INFO] loading encodings...")
with open("./encode.pickle", "rb") as f:
    data = pickle.load(f)

# Preprocessing cho FaceNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Lấy tất cả các đường dẫn ảnh từ dataset
dataset_dir = "./dataset"
imagePaths = list(paths.list_images(dataset_dir))

# Duyệt qua các ảnh trong dataset
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i+1}/{len(imagePaths)}: {imagePath}")

    # Đọc ảnh
    frame = cv2.imread(imagePath)

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
        if min_distance < 0.9:  # Ngưỡng khoảng cách
            matchedIdx = matches.index(min_distance)
            name = data['names'][matchedIdx]

        # Lưu lại tên khuôn mặt
        names.append(name)

        # Vẽ bounding box và tên lên khung hình
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        y = y_min - 15 if y_min - 15 > 15 else y_min + 15
        cv2.putText(frame, name, (x_min, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Tạo thư mục cho nhãn nếu chưa tồn tại
        output_dir = os.path.join('results', name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Lưu ảnh đã phát hiện vào thư mục nhãn
        filename = os.path.basename(imagePath)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)
        print(f"[INFO] Image saved to {output_path}")

print("[INFO] Processing complete.")
