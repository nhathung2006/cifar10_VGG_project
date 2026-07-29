# Phân loại CIFAR-10 bằng VGG16-BN và PyTorch

Project sử dụng VGG16-BN được điều chỉnh cho ảnh CIFAR-10 kích thước `32 x 32`, với 10 lớp đầu ra. Quy trình gồm:

- Chia 50.000 ảnh train thành 45.000 ảnh train và 5.000 ảnh validation.
- Giữ nguyên 10.000 ảnh test.
- Data augmentation cho tập train.
- Batch normalization trong các convolution block.
- Giảm learning rate khi validation loss không cải thiện và early stopping.
- Lưu checkpoint tốt nhất, lịch sử train, biểu đồ và kết quả đánh giá.
- Đánh giá accuracy tổng, accuracy từng lớp và confusion matrix.
- Dự đoán một ảnh bên ngoài với xác suất và `--top-k`.

## 1. Cấu trúc thư mục

```text
cifar10_VGG_project/
├── config.py
├── data.py
├── model.py
├── engine.py
├── utils.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── data/
│   └── cifar-10-batches-py/
└── outputs/
```

Thư mục `data/cifar-10-batches-py/` phải chứa bộ CIFAR-10 dạng gốc gồm `batches.meta`, `data_batch_1` đến `data_batch_5` và `test_batch`.

## 2. Ý nghĩa từng file

| File | Chức năng |
|---|---|
| `config.py` | Lưu cấu hình, đường dẫn, batch size, epoch, learning rate và tên lớp |
| `data.py` | Đọc dữ liệu cục bộ, augmentation, chia train/validation và tạo DataLoader |
| `model.py` | Khai báo mô hình VGG16-BN cho CIFAR-10 |
| `engine.py` | Chứa vòng lặp train và evaluate |
| `utils.py` | Seed, thiết bị, checkpoint, biểu đồ và file kết quả |
| `train.py` | Huấn luyện mô hình |
| `evaluate.py` | Đánh giá checkpoint tốt nhất trên test set |
| `predict.py` | Dự đoán một ảnh bên ngoài |

## 3. Tạo môi trường ảo trên Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chuẩn bị dữ liệu

Dữ liệu CIFAR-10 đã được tải từ Kaggle và giải nén theo đúng cấu trúc sau:

```text
data/
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    └── test_batch
```

Project chỉ đọc dữ liệu cục bộ bằng `download=False`; chương trình không tự tải dữ liệu khi chạy.

## 5. Kiểm tra cấu trúc mô hình

```powershell
python model.py
```

Mô hình nhận tensor `3 x 32 x 32` và trả về 10 logits cho mỗi ảnh.

## 6. Huấn luyện

```powershell
python train.py
```

Checkpoint có validation loss tốt nhất được lưu tại:

```text
outputs/best_vgg16_cifar10.pt
```

Các chỉ số và biểu đồ chỉ được tạo sau khi quá trình train thực sự chạy. README này không giả định trước kết quả accuracy.

## 7. Đánh giá test set

```powershell
python evaluate.py
```

Kết quả được lưu sau khi đánh giá vào:

```text
outputs/test_metrics.json
outputs/confusion_matrix.png
```

## 8. Dự đoán một ảnh

```powershell
python predict.py --image "duong_dan_anh.png"
```

Có thể thay đổi số lượng kết quả hiển thị:

```powershell
python predict.py --image "duong_dan_anh.png" --top-k 5
```

Ảnh được resize về `32 x 32`, chuyển sang tensor và normalize giống dữ liệu đánh giá.

## 9. Các file được tạo sau khi train và evaluate

```text
outputs/
├── best_vgg16_cifar10.pt
├── training_history.csv
├── loss_history.png
├── accuracy_history.png
├── test_metrics.json
└── confusion_matrix.png
```

## 10. Cấu trúc VGG16-BN cho CIFAR-10

VGG16-BN gồm 5 convolution block. Mỗi convolution dùng `3 x 3`, kèm BatchNorm và ReLU; cuối mỗi block là MaxPool `2 x 2`.

```text
Input: 3 × 32 × 32
  ↓
Block 1: Conv-BN-ReLU 64, Conv-BN-ReLU 64, MaxPool
  ↓
Block 2: Conv-BN-ReLU 128, Conv-BN-ReLU 128, MaxPool
  ↓
Block 3: Conv-BN-ReLU 256 × 3, MaxPool
  ↓
Block 4: Conv-BN-ReLU 512 × 3, MaxPool
  ↓
Block 5: Conv-BN-ReLU 512 × 3, MaxPool
  ↓
Classifier với dropout 0.5
  ↓
10 logits
```

## 11. Ghi chú

- Không thêm `Softmax` vào đầu ra mô hình khi train vì project dùng `CrossEntropyLoss`.
- `Softmax` chỉ được dùng trong `predict.py` để hiển thị xác suất.
- Khi chạy trên Windows và gặp lỗi DataLoader, có thể đổi `NUM_WORKERS = 0` trong `config.py`.
