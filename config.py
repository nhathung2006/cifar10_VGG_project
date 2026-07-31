from pathlib import Path

# Đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path("/kaggle/working/cifar10_data/cifar10")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "best_vgg16_cifar10.pt"
HISTORY_CSV_PATH = OUTPUT_DIR / "training_history.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_history.png"
ACCURACY_PLOT_PATH = OUTPUT_DIR / "accuracy_history.png"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

# Cấu hình dữ liệu
SEED = 42
VAL_RATIO = 0.10
BATCH_SIZE = 128
NUM_WORKERS = 2

# Cấu hình mô hình
NUM_CLASSES = 10
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Chuẩn hóa theo cách đơn giản, phổ biến trong tutorial CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Cấu hình huấn luyện
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Giảm learning rate khi validation loss không cải thiện
LR_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LEARNING_RATE = 1e-6

# Dừng sớm khi validation loss không cải thiện
EARLY_STOPPING_PATIENCE = 17
