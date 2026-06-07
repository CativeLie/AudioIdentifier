import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.signal import resample
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
MLP_TARGET_LEN = 4000          # downsample 66150 to 4000 (~16.5x factor)
CNN_TARGET_FRAMES = 130        # mel time frames: 66150/512 ≈ 130
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


class Tee:
    """Duplicate stdout to console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]
EXCLUDED_CLASSES = set()

# Map original class IDs -> contiguous 0..N-1
_LABEL_MAP = {i: i for i in range(10)}

DATA_1D_DIR = os.path.join(BASE_DIR, "save_1d")
DATA_MEL_DIR = os.path.join(BASE_DIR, "save_mel_npy")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")


# ── Data Augmentation ─────────────────────────────────
def add_noise(waveform, noise_level=0.005):
    """Inject Gaussian noise into 1D waveform"""
    noise = np.random.randn(len(waveform)).astype(np.float32) * noise_level
    return waveform + noise

def random_gain(waveform, db_range=(-6, 6)):
    """Random volume change in dB"""
    db = np.random.uniform(*db_range)
    return waveform * (10 ** (db / 20))

def specaugment(mel, freq_mask_param=16, time_mask_param=16, n_freq=2, n_time=2):
    """
    SpecAugment: mask random frequency bands and time steps on mel spectrogram.
    Forces the model to rely on distributed features, improving robustness
    to different recording conditions.
    """
    mel = mel.numpy()  # (1, H, W)
    _, H, W = mel.shape
    # Frequency masking
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, H - f)
        mel[:, f0:f0 + f, :] = 0
    # Time masking
    for _ in range(n_time):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, W - t)
        mel[:, :, t0:t0 + t] = 0
    return torch.tensor(mel, dtype=torch.float32)


# ── Datasets ──────────────────────────────────────────
class MLPDataset(Dataset):
    """MLP dataset: downsample 1D waveform to fixed length"""
    def __init__(self, file_paths, labels, target_len=MLP_TARGET_LEN, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.target_len = target_len
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        y = np.load(self.file_paths[idx]).astype(np.float32)
        if len(y) != self.target_len:
            y = resample(y, self.target_len)
        if self.augment:
            y = random_gain(y)
            y = add_noise(y)
        y = (y - y.mean()) / (y.std() + 1e-8)  # z-score normalize
        return torch.tensor(y, dtype=torch.float32), self.labels[idx]


class CNNDataset(Dataset):
    """CNN dataset: load pre-computed mel spectrograms (128, 130)"""
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        x = np.load(self.file_paths[idx]).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = torch.tensor(x).unsqueeze(0)  # (1, 128, 130)
        if self.augment:
            x = specaugment(x)
        return x, self.labels[idx]


# ── Data Loading ──────────────────────────────────────
def load_data():
    files_1d = sorted([os.path.join(DATA_1D_DIR, f)
                        for f in os.listdir(DATA_1D_DIR) if f.endswith(".npy")])
    files_mel = sorted([os.path.join(DATA_MEL_DIR, f)
                         for f in os.listdir(DATA_MEL_DIR) if f.endswith(".npy")])

    def extract_label(fpath):
        orig = int(os.path.basename(fpath).split("-")[1])
        return _LABEL_MAP[orig]

    # Filter out excluded classes
    files_1d  = [f for f in files_1d  if int(os.path.basename(f).split("-")[1]) not in EXCLUDED_CLASSES]
    files_mel = [f for f in files_mel if int(os.path.basename(f).split("-")[1]) not in EXCLUDED_CLASSES]

    labels_1d = [extract_label(f) for f in files_1d]
    labels_mel = [extract_label(f) for f in files_mel]

    train_1d, val_1d, yt_1d, yv_1d = train_test_split(
        files_1d, labels_1d, test_size=0.2, random_state=42, stratify=labels_1d)
    train_mel, val_mel, yt_mel, yv_mel = train_test_split(
        files_mel, labels_mel, test_size=0.2, random_state=42, stratify=labels_mel)

    print(f"MLP — Train: {len(train_1d)}, Val: {len(val_1d)}")
    print(f"CNN — Train: {len(train_mel)}, Val: {len(val_mel)}")

    train_loader_1d = DataLoader(MLPDataset(train_1d, yt_1d, augment=True),
                                 batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_1d   = DataLoader(MLPDataset(val_1d, yv_1d, augment=False),
                                 batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_loader_2d = DataLoader(CNNDataset(train_mel, yt_mel, augment=True),
                                 batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_2d   = DataLoader(CNNDataset(val_mel, yv_mel, augment=False),
                                 batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return (train_loader_1d, val_loader_1d), (train_loader_2d, val_loader_2d)


# ── MLP Model ─────────────────────────────────────────
class MLP(nn.Module):
    """Multi-layer perceptron: input = downsampled 1D waveform (~4000 dim)"""
    def __init__(self, input_dim=MLP_TARGET_LEN, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ── CNN Model ─────────────────────────────────────────
class CNN(nn.Module):
    """Convolutional neural network: input = mel spectrogram (1, 128, 130)"""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> (32, 64, 65)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> (64, 32, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> (128, 16, 16)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # -> (256, 8, 8)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ── Training & Evaluation ─────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train_model(model, train_loader, val_loader, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_acc, patience_cnt = 0, 0
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        hist["train_loss"].append(train_loss)
        hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        scheduler.step(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"{time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"{model_name}_best.pth"))
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 8:
                print(f"Early stopping (epoch {epoch})")
                break

    print(f"{model_name} best val accuracy: {best_acc:.4f}")
    return hist, best_acc


# ── Plotting ──────────────────────────────────────────
def plot_comparison(mlp_h, cnn_h):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MLP vs CNN — Urban Sound Classification", fontsize=14, fontweight="bold")

    # Loss curves
    ax = axes[0, 0]
    ax.plot(mlp_h["train_loss"], "b--", alpha=0.6, label="MLP Train")
    ax.plot(mlp_h["val_loss"], "b-", label="MLP Val")
    ax.plot(cnn_h["train_loss"], "r--", alpha=0.6, label="CNN Train")
    ax.plot(cnn_h["val_loss"], "r-", label="CNN Val")
    ax.set_title("Loss Curves")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(mlp_h["train_acc"], "b--", alpha=0.6, label="MLP Train")
    ax.plot(mlp_h["val_acc"], "b-", label="MLP Val")
    ax.plot(cnn_h["train_acc"], "r--", alpha=0.6, label="CNN Train")
    ax.plot(cnn_h["val_acc"], "r-", label="CNN Val")
    ax.set_title("Accuracy Curves")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Final accuracy bar
    ax = axes[1, 0]
    mlp_final = mlp_h["val_acc"][-1]
    cnn_final = cnn_h["val_acc"][-1]
    bars = ax.bar(["MLP (1D Waveform)", "CNN (Mel Spectrogram)"], [mlp_final, cnn_final],
                  color=["steelblue", "darkorange"], edgecolor="black")
    ax.set_title("Final Validation Accuracy")
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, [mlp_final, cnn_final]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontweight="bold", fontsize=13)

    # Info text
    ax = axes[1, 1]
    ax.axis("off")
    info = (
        "Model Architecture:\n\n"
        f"MLP: {sum(p.numel() for p in MLP().parameters()):,} params\n"
        "  Input: downsampled 1D waveform (4000-dim)\n"
        "  Layers: FC + BatchNorm + Dropout\n\n"
        f"CNN: {sum(p.numel() for p in CNN().parameters()):,} params\n"
        "  Input: mel spectrogram (128x130)\n"
        "  Layers: Conv2D + BatchNorm + MaxPool -> FC"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/training_comparison.png")


def plot_confusion(model, loader, model_name):
    model.eval()
    _, _, preds, labels = evaluate(model, loader, nn.CrossEntropyLoss())
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion.png"), dpi=150)
    plt.close()
    print(f"Saved: plots/{model_name}_confusion.png")


# ── Main ──────────────────────────────────────────────
def main():
    print("Loading data...")
    (train_1d, val_1d), (train_2d, val_2d) = load_data()

    # Train MLP
    mlp = MLP()
    print(f"\nMLP params: {sum(p.numel() for p in mlp.parameters()):,}")
    mlp_hist, mlp_best = train_model(mlp, train_1d, val_1d, "MLP")

    # Train CNN
    cnn = CNN()
    print(f"\nCNN params: {sum(p.numel() for p in cnn.parameters()):,}")
    cnn_hist, cnn_best = train_model(cnn, train_2d, val_2d, "CNN")

    # ── Final Evaluation ──
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    mlp.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "MLP_best.pth"), weights_only=True))
    _, mlp_acc, mlp_preds, mlp_labels = evaluate(
        mlp, val_1d, nn.CrossEntropyLoss())
    mlp_f1 = f1_score(mlp_labels, mlp_preds, average="weighted")

    cnn.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "CNN_best.pth"), weights_only=True))
    _, cnn_acc, cnn_preds, cnn_labels = evaluate(
        cnn, val_2d, nn.CrossEntropyLoss())
    cnn_f1 = f1_score(cnn_labels, cnn_preds, average="weighted")

    print(f"\nMLP -> Accuracy: {mlp_acc:.4f}  |  Weighted F1: {mlp_f1:.4f}")
    print(f"CNN -> Accuracy: {cnn_acc:.4f}  |  Weighted F1: {cnn_f1:.4f}")

    print("\n--- MLP Classification Report ---")
    print(classification_report(mlp_labels, mlp_preds, target_names=CLASS_NAMES))
    print("\n--- CNN Classification Report ---")
    print(classification_report(cnn_labels, cnn_preds, target_names=CLASS_NAMES))

    # ── Plots ──
    plot_comparison(mlp_hist, cnn_hist)
    plot_confusion(mlp, val_1d, "MLP")
    plot_confusion(cnn, val_2d, "CNN")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'MLP':<12} {'CNN':<12}")
    print("-" * 46)
    print(f"{'Val Accuracy':<20} {mlp_acc:<12.4f} {cnn_acc:<12.4f}")
    print(f"{'Weighted F1':<20} {mlp_f1:<12.4f} {cnn_f1:<12.4f}")
    print(f"{'Best Val Acc':<20} {mlp_best:<12.4f} {cnn_best:<12.4f}")

    # Analysis
    print("\n--- Analysis ---")
    if cnn_acc > mlp_acc:
        diff = cnn_acc - mlp_acc
        print(f"CNN outperforms MLP by {diff:.2%}.")
    else:
        diff = mlp_acc - cnn_acc
        print(f"MLP outperforms CNN by {diff:.2%}. Possible reason: limited data or "
              "CNN needs further hyperparameter tuning.")

    print("\nDone! Models saved in 'models/', plots saved in 'plots/'.")


if __name__ == "__main__":
    log_path = os.path.join(LOGS_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = Tee(log_path)
    print(f"Logging to: {log_path}\n")
    main()
    sys.stdout.log.close()
