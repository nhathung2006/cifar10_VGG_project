# Phân loại CIFAR-10 bằng LeNet và PyTorch

Project hoàn chỉnh gồm:

- Tự động tải CIFAR-10.
- Chia 50.000 ảnh train gốc thành:
  - 45.000 ảnh train.
  - 5.000 ảnh validation.
- Giữ nguyên 10.000 ảnh test.
- Data augmentation cho tập train.
- Huấn luyện LeNet.
- Tự động giảm learning rate khi validation loss không cải thiện.
- Early stopping.
- Lưu checkpoint tốt nhất.
- Vẽ loss và accuracy.
- Đánh giá accuracy tổng, accuracy từng lớp và confusion matrix.
- Dự đoán một ảnh bên ngoài.

## 1. Cấu trúc thư mục

```text
cifar10_lenet_project/
├── config.py
├── data.py
├── model.py
├── engine.py
├── utils.py
├── train.py
├── evaluate.py
├── predict.py
├── export_sample.py
├── requirements.txt
├── data/
└── outputs/
```

## 2. Ý nghĩa từng file

| File | Chức năng |
|---|---|
| `config.py` | Lưu đường dẫn, batch size, epoch, learning rate và tên lớp |
| `data.py` | Tải dữ liệu, augmentation, chia train/validation, tạo DataLoader |
| `model.py` | Khai báo mô hình LeNet |
| `engine.py` | Chứa vòng lặp train và evaluate |
| `utils.py` | Seed, thiết bị, checkpoint, biểu đồ và file kết quả |
| `train.py` | Huấn luyện toàn bộ mô hình |
| `evaluate.py` | Đánh giá model tốt nhất trên test set |
| `predict.py` | Dự đoán một ảnh bên ngoài |
| `export_sample.py` | Xuất một ảnh test mẫu để thử dự đoán |

## 3. Tạo môi trường ảo trên Windows

Mở Terminal tại thư mục project:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu bạn đã cài PyTorch đúng theo CUDA của máy, chỉ cần cài các thư viện còn thiếu.

## 4. Kiểm tra cấu trúc mô hình

```powershell
python model.py
```

Đầu ra mong đợi:

```text
Input shape : (8, 3, 32, 32)
Output shape: (8, 10)
```

## 5. Train mô hình

```powershell
python train.py
```

Lần chạy đầu tiên, `torchvision` sẽ tự tải CIFAR-10 vào thư mục `data/`.

Trong quá trình train, chương trình in:

```text
Epoch 01/30 | LR: ... | Train loss: ... | Train acc: ... | Val loss: ... | Val acc: ...
```

Model có validation loss tốt nhất được lưu tại:

```text
outputs/best_lenet_cifar10.pt
```

## 6. Đánh giá test set

```powershell
python evaluate.py
```

Chương trình in:

- Test loss.
- Test accuracy.
- Accuracy của từng lớp.

Đồng thời tạo:

```text
outputs/confusion_matrix.png
outputs/test_metrics.json
```

## 7. Dự đoán một ảnh

Trước tiên có thể xuất một ảnh test mẫu:

```powershell
python export_sample.py
```

Sau đó chạy:

```powershell
python predict.py --image "outputs/sample_cat.png"
```

Tên file thực tế phụ thuộc ảnh đầu tiên được xuất.

Có thể hiển thị năm dự đoán cao nhất:

```powershell
python predict.py --image "duong_dan_anh.png" --top-k 5
```

## 8. Các file được tạo sau khi train

```text
outputs/
├── best_lenet_cifar10.pt
├── training_history.csv
├── loss_history.png
├── accuracy_history.png
├── test_metrics.json
└── confusion_matrix.png
```

## 9. Cấu trúc LeNet trong project

```text
Input: 3 × 32 × 32
  ↓
Conv2d 3→6, kernel 5×5
  ↓
ReLU
  ↓
MaxPool2d 2×2
  ↓
6 × 14 × 14
  ↓
Conv2d 6→16, kernel 5×5
  ↓
ReLU
  ↓
MaxPool2d 2×2
  ↓
16 × 5 × 5
  ↓
Flatten = 400
  ↓
Linear 400→120
  ↓
ReLU
  ↓
Linear 120→84
  ↓
ReLU
  ↓
Linear 84→10
```

## 10. Ghi chú

- Không thêm `Softmax` ở lớp cuối khi train vì project dùng `CrossEntropyLoss`.
- `Softmax` chỉ được dùng trong `predict.py` để hiển thị xác suất.
- LeNet là mô hình nhỏ, phù hợp để học CNN nhưng không phải kiến trúc mạnh nhất cho CIFAR-10.
- Khi chạy trên Windows và gặp lỗi DataLoader, đổi `NUM_WORKERS = 0` trong `config.py`.
