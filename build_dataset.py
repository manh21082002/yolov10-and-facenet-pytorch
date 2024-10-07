import argparse
import cv2
import os

# Khởi tạo đối số dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())

# Hàm để khởi động webcam và chụp ảnh
def capture_photos(person_name):
    video = cv2.VideoCapture(0)
    total = 0
    print(f"[INFO] Bắt đầu chụp ảnh cho {person_name}...")

    while True:
        # Đọc khung hình từ webcam
        ret, frame = video.read()

        # Hiển thị khung hình lên màn hình
        cv2.imshow("video", frame)
        key = cv2.waitKey(1) & 0xFF

        # Nhấn 'k' để chụp và lưu ảnh
        if key == ord("k"):
            # Tạo thư mục mới cho mỗi người nếu chưa tồn tại
            person_dir = os.path.sep.join([args["output"], person_name])
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

            # Tạo đường dẫn cho ảnh và lưu ảnh
            p = os.path.sep.join([person_dir, "{}.png".format(str(total).zfill(5))])  # Đặt tên file ảnh
            cv2.imwrite(p, frame)
            print(f"Đã lưu ảnh: {p}")
            total += 1

        # Nhấn 'n' để chuyển sang người mới, lưu ảnh của người hiện tại
        elif key == ord("n"):
            print(f"[INFO] Dừng chụp ảnh cho {person_name}.")
            video.release()
            cv2.destroyAllWindows()  # Đóng cửa sổ camera trước khi nhập tên người mới
            return 'new_person'

        # Nhấn 'q' để thoát chương trình
        elif key == ord("q"):
            print(f"[INFO] Dừng chụp ảnh cho {person_name}.")
            video.release()
            cv2.destroyAllWindows()  # Đóng tất cả cửa sổ
            return 'quit'

# Bắt đầu với việc nhập tên người đầu tiên
current_name = input("Nhập tên người đầu tiên: ").strip()

while True:
    # Gọi hàm chụp ảnh
    action = capture_photos(current_name)

    # Kiểm tra hành động sau khi chụp xong
    if action == 'new_person':
        current_name = input("Nhập tên người mới: ").strip()  # Nhập tên người mới
    elif action == 'quit':
        print("[INFO] Chương trình đã kết thúc.")
        break
