import cv2
import os
from datetime import datetime

# Tạo thư mục nếu chưa có
if not os.path.exists('/content/DeepPixBis/spoof'):
    os.makedirs('/content/DeepPixBis/spoof')
if not os.path.exists('/content/DeepPixBis/live'):
    os.makedirs('/content/DeepPixBis/live')

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam!")
    exit()

print("Nhấn 's' để lưu ảnh vào thư mục spoof.")
print("Nhấn 'l' để lưu ảnh vào thư mục live.")
print("Nhấn 'q' để thoát chương trình.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình!")
        break

    # Hiển thị khung hình
    cv2.imshow('Webcam - Nhấn s (spoof), l (live), q (thoát)', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f'captured/spoof/spoof_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, frame)
        print(f'Đã lưu {filename}')

    elif key == ord('l'):
        filename = f'captured/live/live_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, frame)
        print(f'Đã lưu {filename}')

    elif key == ord('q'):
        print('Đã thoát chương trình.')
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()
