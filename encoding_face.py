from imutils import paths
import argparse
import pickle
import cv2
import os
import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the directory of faces and images")
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
args = vars(ap.parse_args())

# Load YOLO model
print("[INFO] loading YOLO model for face detection...")
yolo_model = YOLO("D:/AI_programing_lastterm/weight/best.pt")

# Load FaceNet (InceptionResnetV1)
print("[INFO] loading FaceNet model for face recognition...")
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing pipeline cho FaceNet
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),  # Kích thước yêu cầu cho FaceNet
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Lấy paths của các ảnh trong dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# Khởi tạo lists chứa các encodings và names
knownEncodings = []
knownNames = []

# Duyệt qua các image paths
for (i, imagePath) in enumerate(imagePaths):
    # Lấy tên người từ imagePath
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # Load ảnh và phát hiện khuôn mặt bằng YOLO
    image = cv2.imread(imagePath)
    results = yolo_model(image)

    # Nếu không phát hiện khuôn mặt, bỏ qua ảnh này
    if len(results[0].boxes) == 0:
        print("[INFO] No face detected in image, skipping...")
        continue

    # Duyệt qua các bounding box phát hiện được và tính toán embeddings
    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0].cpu().numpy())

        # Cắt khuôn mặt ra khỏi khung hình
        face_image = image[y_min:y_max, x_min:x_max]

        # Chuyển đổi ảnh về tensor và tính embedding bằng FaceNet
        face_tensor = preprocess(face_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            encoding = facenet_model(face_tensor).cpu().numpy()

        # Lưu encoding và tên vào lists
        knownEncodings.append(encoding)
        knownNames.append(name)

# Lưu các encodings và names vào file
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}

with open(args["encodings"], "ab") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encodings successfully serialized to", args["encodings"])
