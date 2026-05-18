# statistics/task1_model_check.py
"""
Task 1 模型最终对照：
固定特征为 G 配置（v2 + pitch, 341 维）
对比：
  M1. LR + 不加 class_weight    （主推荐）
  M2. LR + class_weight=balanced (验证之前的猜测)
  M3. LR + class_weight=balanced + 调小 C（更强正则化）
  M4. RandomForest               (验证树模型是否更稳)
  M5. LR (M1) + RF (M4) 集成     (看融合是否提升)
"""
import os
import numpy as np
import miditoolkit
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

DATAROOT = "student_files/task1_composer_classification"
OUTPUT_FILE = "statistics/task1_model_check.txt"

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 特征提取（复制自 task1_feature_v2.py）
# ============================================================
PITCH_MIN, PITCH_MAX = 21, 108
N_PITCHES = PITCH_MAX - PITCH_MIN + 1
INTERVAL_MIN, INTERVAL_MAX = -12, 12
N_INTERVALS = INTERVAL_MAX - INTERVAL_MIN + 1


def compute_polyphony(notes):
    events = []
    for n in notes:
        events.append((n.start, 1)); events.append((n.end, -1))
    events.sort()
    cur = max_p = 0
    for _, d in events:
        cur += d; max_p = max(max_p, cur)
    return max_p


def interval_hist(pitch_sequence):
    if len(pitch_sequence) < 2:
        return np.zeros(N_INTERVALS, dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ih = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX):
        ih[iv - INTERVAL_MIN] += 1
    ih /= ih.sum() + 1e-9
    return ih


def interval_transition_mod12(pitch_sequence):
    if len(pitch_sequence) < 3:
        return np.zeros((12, 12), dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ic = intervals % 12
    tr = np.zeros((12, 12), dtype=np.float32)
    for i in range(len(ic) - 1):
        tr[ic[i], ic[i+1]] += 1
    return tr


def ioi_hist(start_sequence):
    if len(start_sequence) < 2:
        return np.zeros(12, dtype=np.float32)
    iois = np.diff(np.array(start_sequence))
    iois = iois[iois > 0]
    if len(iois) == 0:
        return np.zeros(12, dtype=np.float32)
    il = np.log2(np.maximum(iois, 1))
    h, _ = np.histogram(il, bins=12, range=(0, 14))
    h = h.astype(np.float32) / (h.sum() + 1e-9)
    return h


def extract_G_features(midi_path):
    """直接拼接 G 配置（341 维）"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    if len(notes) < 2:
        return np.zeros(341, dtype=np.float32)
    
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    events = defaultdict(list)
    for n in notes:
        events[n.start].append(n)
    sorted_starts = sorted(events.keys())
    melody_pitches = [max(nn.pitch for nn in events[s]) for s in sorted_starts]
    bass_pitches = [min(nn.pitch for nn in events[s]) for s in sorted_starts]
    
    # pitch (88)
    ph = np.zeros(N_PITCHES, dtype=np.float32)
    for p in np.clip(pitches, PITCH_MIN, PITCH_MAX):
        ph[p - PITCH_MIN] += 1
    ph /= ph.sum() + 1e-9
    
    # chroma (12)
    ch = np.zeros(12, dtype=np.float32)
    for p in pitches:
        ch[p % 12] += 1
    ch /= ch.sum() + 1e-9
    
    # interval_melody (25), interval_bass (25)
    im = interval_hist(melody_pitches)
    ib = interval_hist(bass_pitches)
    
    # transition_int12 (144) = (melody trans + bass trans), row-normalized
    tr_m = interval_transition_mod12(melody_pitches)
    tr_b = interval_transition_mod12(bass_pitches)
    tr = tr_m + tr_b
    tr = tr / (tr.sum(axis=1, keepdims=True) + 1e-9)
    
    # duration (12)
    dl = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dl, bins=12, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    
    # velocity_hist (12)
    vh, _ = np.histogram(velocities, bins=12, range=(0, 128))
    vh = vh.astype(np.float32) / (vh.sum() + 1e-9)
    
    # ioi_new (12)
    ih2 = ioi_hist(sorted_starts)
    
    # scalars (11)
    legato_ratios = []
    for i in range(len(sorted_starts) - 1):
        s_cur, s_next = sorted_starts[i], sorted_starts[i + 1]
        gap = s_next - s_cur
        if gap > 0:
            max_dur = max(nn.end - nn.start for nn in events[s_cur])
            legato_ratios.append(min(max_dur / gap, 2.0))
    legato_mean = float(np.mean(legato_ratios)) if legato_ratios else 0.0
    legato_std = float(np.std(legato_ratios)) if legato_ratios else 0.0
    
    total_ticks = max(starts.max() - starts.min(), 1)
    scalars = np.array([
        np.log1p(len(notes)), np.log1p(total_ticks / 480.0),
        len(notes) / (total_ticks / 480.0),
        float(pitches.max() - pitches.min()) / 88,
        float(pitches.std()) / 24,
        float(pitches.mean()) / 127,
        float(velocities.mean()) / 127,
        float(velocities.std()) / 64,