# CAS\_TBM - Face Anti-Spoofing using DeepPixBis

## Mô tả

Đây là project sử dụng mô hình **DeepPixBis** để phát hiện ảnh giả mạo (spoof) và ảnh thật (live) trong bài toán chống giả mạo khuôn mặt.
Project này thực hiện quá trình:

* Tiền xử lý ảnh
* Cắt khuôn mặt
* Huấn luyện mô hình DeepPixBis
* Đánh giá mô hình trên tập kiểm thử

## Hướng dẫn cài đặt

```bash
git clone https://github.com/your-username/deeppixbis.git
cd deeppixbis
pip install -r requirements.txt
```

*Hoặc cài tay nếu thiếu thư viện:*

```bash
pip install facenet-pytorch imutils scikit-learn opencv-python
```

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

* Tạo hai thư mục:

  * `/live`: chứa ảnh khuôn mặt thật
  * `/spoof`: chứa ảnh khuôn mặt giả mạo

> **Lưu ý:** Đường dẫn trong file `main.py` đang mặc định cho Colab. Nếu chạy trên máy cá nhân, hãy đổi lại đường dẫn phù hợp.

### 2. Chạy mô hình

* Nếu muốn **huấn luyện lại mô hình**, mở `main.py` và chỉnh:

```python
TRAIN = True
```

* Nếu muốn **dùng mô hình đã huấn luyện**, giữ nguyên:

```python
TRAIN = False
```

* Chạy file:

```bash
python main.py
```

### 3. Kết quả

* In ra **độ chính xác (accuracy)** và **báo cáo phân loại (classification report)**
* Hiển thị kết quả dự đoán của từng ảnh:

  * Tên file
  * Dự đoán: Live hoặc Spoof
  * Xác suất dự đoán (confidence)

### 4. Chụp ảnh với webcam

Bạn có thể sử dụng script:

```bash
python capture_webcam.py
```

> Dùng để lưu ảnh trực tiếp vào thư mục `/live` hoặc `/spoof`.

## Ghi chú

* Google Colab **không hỗ trợ webcam trực tiếp.** Nếu cần chụp ảnh, bạn phải chạy trên máy tính cá nhân.
* Nếu gặp lỗi `git push rejected`, bạn có thể dùng `git push -f` để force push.

## Tác giả

* Gốc: [Abhishek Bhardwaj](https://github.com/AbhishekBhardwaj123)
* Fork và phát triển thêm: [Anh Lưu]([https://github.com/your-username](https://github.com/anhluufromVietnam/deeppixbis/))

---
