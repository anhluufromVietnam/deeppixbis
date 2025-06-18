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
git clone https://github.com/anhluufromVietnam/deeppixbis.git
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
* Hướng dẫn chi tiết hơn ở phần tiếng Anh.

## Tác giả

* Gốc: [Abhishek Bhardwaj](https://github.com/AbhishekBhardwaj123)
* Fork và phát triển thêm: [Anh Lưu](https://github.com/anhluufromVietnam/deeppixbis/)

---

## 📌 Description

This project applies the **DeepPixBis** model to detect **spoof (fake)** and **live (genuine)** face images in face anti-spoofing tasks.

The pipeline includes:

* Image Preprocessing
* Face Cropping
* DeepPixBis Model Training
* Model Evaluation on Test Set
* Visualization of Predictions and Confidence

---

## 📂 Dataset Structure

Organize your dataset as follows:

```
DeepPixBis/
│
├── live/     # Genuine face images
│   ├── live1.jpg
│   ├── live2.jpg
│   └── ...
│
└── spoof/    # Spoof face images
    ├── spoof1.jpg
    ├── spoof2.jpg
    └── ...
```

---

## 🚀 Installation

```bash
git clone https://github.com/anhluufromVietnam/deeppixbis.git
cd deeppixbis
pip install -r requirements.txt
```

Or install manually:

```bash
pip install facenet-pytorch imutils scikit-learn opencv-python
```

---

## ✅ How to Use

### 🔹 Training the Model

In `main.py`, set:

```python
TRAIN = True
```

Then run:

```bash
python main.py
```

### 🔹 Using Pre-trained Model

Keep:

```python
TRAIN = False
```

Then run:

```bash
python main.py
```

### 🔹 Result Example

The script will print the **accuracy** and detailed **classification report**.

Example:

```
File: live1.jpg - Prediction: Live (Confidence: 94.3%)
File: spoof3.jpg - Prediction: Spoof (Confidence: 88.5%)

Classification Accuracy Obtained: 91.5%

Classification Report:
              precision    recall  f1-score   support

       Live       0.92      0.95      0.93       20
      Spoof      0.91      0.85      0.88       20

    accuracy                           0.92       40
```

---

## 🎥 Webcam Capture (Local Machine Only)

You can capture images directly using:

```bash
python capture_webcam.py
```

* Press `l` to capture an live image.
* Press `s` to capture an spoof image.
* Press `q` to quit.
* Images will be saved to `live/` or `spoof/` based on your selection.

> ⚠️ **Google Colab does not support webcam. Run this script on your PC.**

---

## 📸 Sample Result Images (You can insert later)

| Image               | Prediction | Confidence |
| ------------------- | ---------- | ---------- |
| ![](path_to_image1) | Live       | 94.3%      |
| ![](path_to_image2) | Spoof      | 88.5%      |

---

## 🙏 Credits

* Original Author: [Abhishek Bhardwaj](https://github.com/AbhishekBhardwaj123)
* Forked and Extended by: [Anh Lưu](https://github.com/anhluufromVietnam/deeppixbis/)

