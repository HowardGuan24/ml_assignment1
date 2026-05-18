"""
Task 3: multi-label audio tagging.

Default experiment:
- cache full-track log-mel spectrograms on disk
- train a pretrained ImageNet ResNet18 on mel spectrogram crops
- use greedy multi-label folds and average fold predictions
- use AMP, SpecAugment, optional mixup, and multi-crop TTA

Useful server commands:
  TASK3_FOLDS=1 TASK3_EPOCHS=10 python training/task3_train.py
  TASK3_FOLDS=3 TASK3_EPOCHS=30 TASK3_DEVICE=cuda:0 python training/task3_train.py
  TASK3_MODEL=cnn TASK3_FOLDS=3 python training/task3_train.py
"""
import copy
import json
import os
import random
from collections import Counter

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from tqdm import tqdm


DATAROOT = "student_files/task3_audio_classification"
PRED_FILE = "results/predictions3.json"
LOG_FILE = "train_log/task3_results_v2.txt"
CACHE_FILE = "train_log/task3_mel_cache_v2.pt"

TAGS = [
    "rock",
    "oldies",
    "jazz",
    "pop",
    "dance",
    "blues",
    "punk",
    "chill",
    "electronic",
    "country",
]
N_CLASSES = len(TAGS)

SEED = int(os.environ.get("TASK3_SEED", "42"))
MODEL_NAME = os.environ.get("TASK3_MODEL", "resnet18_pretrained")
DEVICE_NAME = os.environ.get("TASK3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE_NAME)

SAMPLE_RATE = int(os.environ.get("TASK3_SAMPLE_RATE", "22050"))
N_MELS = int(os.environ.get("TASK3_N_MELS", "128"))
N_FFT = int(os.environ.get("TASK3_N_FFT", "1024"))
HOP_LENGTH = int(os.environ.get("TASK3_HOP_LENGTH", "512"))
CLIP_SECONDS = float(os.environ.get("TASK3_CLIP_SECONDS", "10"))
TARGET_FRAMES = int(round(CLIP_SECONDS * SAMPLE_RATE / HOP_LENGTH)) + 1

NUM_FOLDS = int(os.environ.get("TASK3_FOLDS", "3"))
NUM_EPOCHS = int(os.environ.get("TASK3_EPOCHS", "35"))
BATCH_SIZE = int(os.environ.get("TASK3_BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("TASK3_NUM_WORKERS", "0"))
LR = float(os.environ.get("TASK3_LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("TASK3_WEIGHT_DECAY", "1e-4"))
PATIENCE = int(os.environ.get("TASK3_PATIENCE", "8"))
MIXUP_ALPHA = float(os.environ.get("TASK3_MIXUP_ALPHA", "0.2"))
MIXUP_PROB = float(os.environ.get("TASK3_MIXUP_PROB", "0.5"))
SPEC_AUG_PROB = float(os.environ.get("TASK3_SPEC_AUG_PROB", "0.8"))
TTA_CROPS = int(os.environ.get("TASK3_TTA_CROPS", "5"))
USE_POS_WEIGHT = os.environ.get("TASK3_POS_WEIGHT", "1") != "0"
USE_AMP = os.environ.get("TASK3_AMP", "1") != "0"
USE_CACHE = os.environ.get("TASK3_CACHE", "1") != "0"

_log = []


def log(message=""):
    print(message)
    _log.append(str(message))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_metadata():
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train_meta = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test_paths = eval(f.read())
    return train_meta, test_paths


def label_vector(tags):
    return torch.tensor([1.0 if tag in tags else 0.0 for tag in TAGS], dtype=torch.float32)


def metadata_to_matrix(meta, paths):
    return np.asarray([[1 if tag in meta[p] else 0 for tag in TAGS] for p in paths], dtype=np.int64)


def greedy_multilabel_folds(meta, n_folds, seed):
    """Approximate iterative stratification without extra dependencies."""
    paths = list(meta.keys())
    rng = random.Random(seed)
    rng.shuffle(paths)
    y = metadata_to_matrix(meta, paths)
    desired = y.sum(axis=0) / n_folds

    fold_counts = np.zeros((n_folds, N_CLASSES), dtype=np.float64)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    folds = [[] for _ in range(n_folds)]

    order = sorted(range(len(paths)), key=lambda i: (int(y[i].sum()), rng.random()), reverse=True)
    for idx in order:
        label = y[idx]
        scores = []
        for fold in range(n_folds):
            new_counts = fold_counts[fold] + label
            label_cost = np.sum((new_counts - desired) ** 2)
            size_cost = 0.05 * (fold_sizes[fold] + 1 - len(paths) / n_folds) ** 2
            scores.append(label_cost + size_cost)
        best = int(np.argmin(scores))
        folds[best].append(paths[idx])
        fold_counts[best] += label
        fold_sizes[best] += 1

    return folds


def make_validation_folds(meta, n_folds, seed):
    if n_folds <= 1:
        paths = list(meta.keys())
        rng = random.Random(seed)
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * 0.1)))
        return [paths[:n_val]]
    return greedy_multilabel_folds(meta, n_folds, seed)


def load_waveform(rel_path):
    wav, _ = librosa.load(os.path.join(DATAROOT, rel_path), sr=SAMPLE_RATE, mono=True)
    if len(wav) == 0:
        wav = np.zeros(int(SAMPLE_RATE * CLIP_SECONDS), dtype=np.float32)
    return torch.from_numpy(wav).float().unsqueeze(0)


def build_mel_cache(paths):
    cache_meta = {
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "version": "task3_mel_cache_v2_full_track",
    }
    if USE_CACHE and os.path.exists(CACHE_FILE):
        try:
            cached = torch.load(CACHE_FILE, map_location="cpu")
            if cached.get("meta") == cache_meta and all(path in cached["features"] for path in paths):
                log(f"Loaded mel cache: {CACHE_FILE}")
                return cached["features"]
            log("Mel cache metadata/path mismatch; rebuilding.")
        except Exception as exc:
            log(f"Could not load mel cache; rebuilding: {type(exc).__name__}: {exc}")

    mel_fn = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=20,
        f_max=SAMPLE_RATE // 2,
        power=2.0,
    )
    db_fn = AmplitudeToDB()
    features = {}
    for path in tqdm(paths, desc="Building mel cache"):
        wav = load_waveform(path)
        mel = db_fn(mel_fn(wav)).squeeze(0).to(torch.float16).contiguous()
        features[path] = mel

    if USE_CACHE:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        torch.save({"meta": cache_meta, "features": features}, CACHE_FILE)
        log(f"Saved mel cache: {CACHE_FILE}")
    return features


def crop_or_pad(mel, start=None, train=False):
    n_frames = mel.shape[-1]
    if n_frames < TARGET_FRAMES:
        mel = F.pad(mel, (0, TARGET_FRAMES - n_frames), value=float(mel.mean()))
        n_frames = TARGET_FRAMES
    if start is None:
        if train and n_frames > TARGET_FRAMES:
            start = random.randint(0, n_frames - TARGET_FRAMES)
        else:
            start = max(0, (n_frames - TARGET_FRAMES) // 2)
    start = max(0, min(int(start), n_frames - TARGET_FRAMES))
    return mel[:, start : start + TARGET_FRAMES]


def normalize_mel(mel):
    mel = mel.float()
    return (mel - mel.mean()) / (mel.std() + 1e-6)


def spec_augment(mel):
    if random.random() > SPEC_AUG_PROB:
        return mel
    out = mel.clone()
    fill = float(out.mean())
    n_mels, n_time = out.shape
    for _ in range(2):
        width = random.randint(0, max(1, n_mels // 5))
        start = random.randint(0, max(0, n_mels - width))
        out[start : start + width, :] = fill
    for _ in range(2):
        width = random.randint(0, max(1, n_time // 8))
        start = random.randint(0, max(0, n_time - width))
        out[:, start : start + width] = fill
    return out


class MelDataset(Dataset):
    def __init__(self, paths, meta, features, train=False):
        self.paths = list(paths)
        self.meta = meta
        self.features = features
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        mel = self.features[path].float()
        mel = crop_or_pad(mel, train=self.train)
        mel = normalize_mel(mel)
        if self.train:
            mel = spec_augment(mel)
        label = label_vector(self.meta.get(path, []))
        return mel.unsqueeze(0), label, path


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class MelResNet(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = ConvBlock(32, 64, stride=2)
        self.stage2 = ConvBlock(64, 128, stride=2)
        self.stage3 = ConvBlock(128, 256, stride=2)
        self.stage4 = ConvBlock(256, 384, stride=2)
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(384 * 2, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        x = torch.cat([avg, mx], dim=1)
        return self.fc(self.dropout(x))


def make_resnet18(pretrained=True):
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        if pretrained and model.conv1.weight.shape[1] == 3:
            w = model.conv1.weight.data.mean(dim=1, keepdim=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.conv1.weight.data.copy_(w)
        else:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(model.fc.in_features, N_CLASSES))
        return model, "resnet18_pretrained" if pretrained else "resnet18_random"
    except Exception as exc:
        log(f"torchvision pretrained ResNet unavailable; using MelResNet: {type(exc).__name__}: {exc}")
        return MelResNet(), "mel_resnet_fallback"


def make_model():
    if MODEL_NAME == "cnn":
        return MelResNet(), "mel_resnet"
    if MODEL_NAME == "resnet18_random":
        return make_resnet18(pretrained=False)
    return make_resnet18(pretrained=True)


def mixup_batch(x, y):
    if MIXUP_ALPHA <= 0 or random.random() > MIXUP_PROB:
        return x, y
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def class_pos_weight(train_meta, train_paths):
    counts = np.zeros(N_CLASSES, dtype=np.float32)
    for path in train_paths:
        for tag in train_meta[path]:
            counts[TAGS.index(tag)] += 1
    neg = len(train_paths) - counts
    pos_weight = neg / np.maximum(counts, 1.0)
    return torch.tensor(np.clip(pos_weight, 1.0, 8.0), dtype=torch.float32)


def evaluate(model, loader):
    model.eval()
    preds, labels, paths = [], [], []
    with torch.no_grad():
        for x, y, batch_paths in loader:
            x = x.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                logits = model(x)
            preds.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(y.numpy())
            paths.extend(list(batch_paths))
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return preds, labels, paths, average_precision_score(labels, preds, average="macro")


def tta_predict(model, paths, meta, features):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for path in tqdm(paths, desc="TTA predict"):
            mel = features[path].float()
            n_frames = mel.shape[-1]
            if n_frames <= TARGET_FRAMES or TTA_CROPS <= 1:
                starts = [max(0, (n_frames - TARGET_FRAMES) // 2)]
            else:
                starts = np.linspace(0, n_frames - TARGET_FRAMES, TTA_CROPS).astype(int).tolist()
            probs = []
            for start in starts:
                crop = normalize_mel(crop_or_pad(mel, start=start, train=False))
                x = crop.unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                    logits = model(x)
                probs.append(torch.sigmoid(logits).float().cpu().numpy()[0])
            all_probs.append(np.mean(probs, axis=0))
    return np.asarray(all_probs, dtype=np.float32), list(paths)


def train_one_fold(fold_idx, train_paths, val_paths, train_meta, features):
    seed_everything(SEED + fold_idx)
    train_set = MelDataset(train_paths, train_meta, features, train=True)
    val_set = MelDataset(val_paths, train_meta, features, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
    )

    model, resolved_name = make_model()
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"\nFold {fold_idx + 1}: model={resolved_name}, params={n_params:,}")

    pos_weight = class_pos_weight(train_meta, train_paths).to(DEVICE) if USE_POS_WEIGHT else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.type == "cuda")

    best_map = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"Fold {fold_idx + 1} epoch {epoch}/{NUM_EPOCHS}"):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            x, y = mixup_batch(x, y)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())

        scheduler.step()
        _, _, _, val_map = evaluate(model, val_loader)
        train_loss = running_loss / max(1, len(train_loader))
        lr_now = scheduler.get_last_lr()[0]
        log(
            f"[Fold {fold_idx + 1} Epoch {epoch}] "
            f"loss={train_loss:.4f} val_mAP={val_map:.4f} lr={lr_now:.6g}"
        )

        if val_map > best_map:
            best_map = val_map
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                log(f"Fold {fold_idx + 1}: early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    log(f"Fold {fold_idx + 1}: best val mAP = {best_map:.4f}")
    return model, best_map


def write_predictions(test_paths, test_probs):
    predictions = {}
    for i, path in enumerate(test_paths):
        key = path[2:] if isinstance(path, str) and path.startswith("./") else path
        predictions[key] = {TAGS[j]: float(test_probs[i, j]) for j in range(N_CLASSES)}
    os.makedirs(os.path.dirname(PRED_FILE), exist_ok=True)
    with open(PRED_FILE, "w") as f:
        f.write(repr(predictions) + "\n")
    log(f"\n[OK] Wrote {PRED_FILE}")
    log(f"Prediction count: {len(predictions)}")


def main():
    seed_everything(SEED)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log("=" * 72)
    log("Task 3 v2: pretrained spectrogram model + fold ensemble")
    log("=" * 72)
    log(
        f"device={DEVICE}, model={MODEL_NAME}, folds={NUM_FOLDS}, epochs={NUM_EPOCHS}, "
        f"batch={BATCH_SIZE}, amp={USE_AMP}, tta_crops={TTA_CROPS}"
    )

    train_meta, test_paths = load_metadata()
    all_train_paths = list(train_meta.keys())
    all_paths = all_train_paths + list(test_paths)
    tag_counts = Counter(tag for tags in train_meta.values() for tag in tags)
    log(f"train={len(train_meta)}, test={len(test_paths)}, tags={dict(tag_counts)}")

    features = build_mel_cache(all_paths)
    folds = make_validation_folds(train_meta, NUM_FOLDS, SEED)
    for i, fold in enumerate(folds):
        y_fold = metadata_to_matrix(train_meta, fold)
        log(f"fold {i + 1}: n={len(fold)}, tag_counts={y_fold.sum(axis=0).tolist()}")

    fold_test_probs = []
    fold_scores = []
    for fold_idx in range(max(1, NUM_FOLDS)):
        val_paths = folds[fold_idx]
        val_set = set(val_paths)
        train_paths = [p for p in all_train_paths if p not in val_set]
        model, best_map = train_one_fold(fold_idx, train_paths, val_paths, train_meta, features)
        fold_scores.append(best_map)
        test_probs, ordered_test_paths = tta_predict(
            model, list(test_paths), {p: [] for p in test_paths}, features
        )
        fold_test_probs.append(test_probs)
        log(f"Fold {fold_idx + 1}: test probability mean={test_probs.mean(axis=0).round(4).tolist()}")

    final_probs = np.mean(fold_test_probs, axis=0)
    write_predictions(ordered_test_paths, final_probs)
    log(f"Fold val mAPs: {[round(x, 4) for x in fold_scores]}")
    log(f"Mean fold val mAP: {float(np.mean(fold_scores)):.4f}")
    log(f"Final test probability mean={final_probs.mean(axis=0).round(4).tolist()}")

    with open(LOG_FILE, "w") as f:
        f.write("\n".join(_log) + "\n")
    log(f"[OK] Wrote {LOG_FILE}")


if __name__ == "__main__":
    main()
