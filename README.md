# YOLOv10 and FaceNet-PyTorch

## Mô tả
Dự án này sử dụng **YOLOv10** để phát hiện khuôn mặt và **FaceNet (InceptionResnetV1)** để nhận diện khuôn mặt. Bạn có thể xây dựng dataset của riêng mình bằng cách chụp ảnh từ webcam và sau đó nhận diện khuôn mặt trong thời gian thực.

## Cách cài đặt và chạy chương trình:

### 1. Tạo môi trường ảo và cài đặt các phụ thuộc:
```bash
# Tạo môi trường ảo
python -m venv env
```
```bash
# Kích hoạt môi trường ảo:
# Trên Windows:
.\env\Scripts\activate
# Trên macOS/Linux:
source env/bin/activate

# Cài đặt các phụ thuộc từ tệp requirements.txt
pip install -r requirements.txt
```

###2. Chạy chương trình tạo dataset:
```bash
python .\build_dataset.py -o .\dataset
```
Hướng dẫn sử dụng:
Sau khi chạy chương trình, nhập tên của người đầu tiên mà bạn muốn chụp ảnh.
- Nhấn k để chụp ảnh và lưu vào thư mục dataset.
- Nhấn n để chuyển sang người mới và nhập tên người mới để tiếp tục chụp ảnh.
- Nhấn q để thoát chương trình.


###3. Chạy chương trình nhận diện khuôn mặt thời gian thực:
Sau khi đã có dataset và đã mã hóa khuôn mặt, chạy chương trình nhận diện khuôn mặt trong thời gian thực từ webcam bằng lệnh sau:

```bash
python .\recognize_faces_realtime.py -e .\encode.pickle
```
Hướng dẫn sử dụng:
- Sau khi chạy chương trình, hệ thống sẽ nhận diện khuôn mặt theo thời gian thực.
- Nhấn q để thoát chương trình.
