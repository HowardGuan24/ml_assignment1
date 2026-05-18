# training/task1_train_v2.py
"""
Task 1 训练 v2:
- 保留 G 配置全部特征 (341 维)
- 新增 metadata 指纹特征 (约 25 维)
"""
import os
import numpy as np
import miditoolkit
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATAROOT = "student_files/task1_composer_classification"
PRED_FILE = "results/predictions1.json"
LOG_FILE = "train_log/task1_results_v2.txt"

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 常量
# ============================================================
PITCH_MIN, PITCH_MAX = 21, 108
N_PITCHES = PITCH_MAX - PITCH_MIN + 1
INTERVAL_MIN, INTERVAL_MAX = -12, 12
N_INTERVALS = INTERVAL_MAX - INTERVAL_MIN + 1

# 所有可能的 key signature (按训练集观察)
ALL_KEYS = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb',
            'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bbm', 'Bm']
N_KEYS = len(ALL_KEYS)
KEY_TO_IDX = {k: i for i, k in enumerate(ALL_KEYS)}


def compute_polyphony(notes):
    events = []
    for n in notes:
        events.append((n.start, 1)); events.append((n.end, -1))
    events.sort()
    cur = max_p = 0
    for _, d in events:
        cur += d; max_p = max(max_p, cur)
    return max_p


def interval_hist(seq):
    if len(seq) < 2:
        return np.zeros(N_INTERVALS, dtype=np.float32)
    intervals = np.diff(np.array(seq))
    ih = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX):
        ih[iv - INTERVAL_MIN] += 1
    ih /= ih.sum() + 1e-9
    return ih


def interval_transition_mod12(seq):
    if len(seq) < 3:
        return np.zeros((12, 12), dtype=np.float32)
    intervals = np.diff(np.array(seq))
    ic = intervals % 12
    tr = np.zeros((12, 12), dtype=np.float32)
    for i in range(len(ic) - 1):
        tr[ic[i], ic[i+1]] += 1
    return tr


def ioi_hist(seq):
    if len(seq) < 2:
        return np.zeros(12, dtype=np.float32)
    iois = np.diff(np.array(seq))
    iois = iois[iois > 0]
    if len(iois) == 0:
        return np.zeros(12, dtype=np.float32)
    il = np.log2(np.maximum(iois, 1))
    h, _ = np.histogram(il, bins=12, range=(0, 14))
    h = h.astype(np.float32) / (h.sum() + 1e-9)
    return h


def extract_metadata_features(midi):
    """提取 metadata 指纹特征 (约 25 维)"""
    feats = []
    
    # 1. first_tempo (1 维)
    if midi.tempo_changes:
        first_tempo = midi.tempo_changes[0].tempo
    else:
        first_tempo = 120.0
    feats.append(first_tempo / 200.0)  # 归一化到约 [0, 2]
    
    # 2. tempo 是否整数（指纹）(1 维)
    feats.append(1.0 if abs(first_tempo - round(first_tempo)) < 1e-6 else 0.0)
    
    # 3. tempo 的小数指纹 (1 维) - 不同软件量化精度不同
    feats.append(first_tempo - int(first_tempo))
    
    # 4. n_tempo_changes (1 维)
    feats.append(min(len(midi.tempo_changes), 10) / 10.0)
    
    # 5. time signature (2 维)
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        feats.append(ts.numerator / 12.0)
        feats.append(ts.denominator / 16.0)
    else:
        feats.append(4 / 12.0); feats.append(4 / 16.0)
    
    # 6. n_time_signature_changes (1 维)
    feats.append(min(len(midi.time_signature_changes), 5) / 5.0)
    
    # 7. key signature one-hot (N_KEYS 维)
    key_oh = [0.0] * N_KEYS
    if midi.key_signature_changes:
        k = midi.key_signature_changes[0].key_name
        if k in KEY_TO_IDX:
            key_oh[KEY_TO_IDX[k]] = 1.0
    feats.extend(key_oh)
    
    # 8. n_key_signature_changes (1 维)
    feats.append(min(len(midi.key_signature_changes), 5) / 5.0)
    
    return np.array(feats, dtype=np.float32)


def extract_features(midi_path):
    """G 配置 (341 维) + metadata 特征 (~38 维)"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    if len(notes) < 2:
        # 防御性
        return np.zeros(341 + 7 + N_KEYS + 1, dtype=np.float32)
    
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
    
    # ----- G 配置部分（不变）-----
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
    
    # interval melody/bass (25+25)
    im = interval_hist(melody_pitches)
    ib = interval_hist(bass_pitches)
    
    # transition mod 12 (144)
    tr_m = interval_transition_mod12(melody_pitches)
    tr_b = interval_transition_mod12(bass_pitches)
    tr = tr_m + tr_b
    tr = tr / (tr.sum(axis=1, keepdims=True) + 1e-9)
    
    # duration (12)
    dl = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dl, bins=12, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    
    # velocity (12)
    vh, _ = np.histogram(velocities, bins=12, range=(0, 128))
    vh = vh.astype(np.float32) / (vh.sum() + 1e-9)
    
    # ioi (12)
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
        compute_polyphony(notes) / 10.0,
        legato_mean, legato_std,
    ], dtype=np.float32)
    
    g_feat = np.concatenate([ph, ch, im, ib, tr.flatten(), dh, vh, ih2, scalars])
    
    # ----- 新增 metadata 特征 -----
    meta_feat = extract_metadata_features(midi)
    
    # ----- 新增更多 velocity / duration 指纹 -----
    extra = np.array([
        min(len(set(velocities.tolist())), 100) / 100.0,  # n_unique_velocities
        min(len(set(durations.tolist())), 50) / 50.0,      # n_unique_durations
        np.log1p(Counter(durations.tolist()).most_common(1)[0][0]),  # log(most_common_duration)
    ], dtype=np.float32)
    
    return np.concatenate([g_feat, meta_feat, extra])


def main():
    log("=" * 70)
    log("Task 1 v2: G 特征 + metadata 指纹")
    log("=" * 70)
    
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test_paths = eval(f.read())
    
    train_paths = list(train.keys())
    y = np.array([train[p] for p in train_paths])
    
    log(f"训练: {len(train_paths)}, 测试: {len(test_paths)}")
    
    log("\n--- 提取训练集特征 ---")
    X_train = np.array(
        [extract_features(os.path.join(DATAROOT, p)) for p in tqdm(train_paths)],
        dtype=np.float32
    )
    log("--- 提取测试集特征 ---")
    X_test = np.array(
        [extract_features(os.path.join(DATAROOT, p)) for p in tqdm(test_paths)],
        dtype=np.float32
    )
    log(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # 5-fold CV
    log("\n" + "=" * 70)
    log("5-fold CV")
    log("=" * 70)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X_train, y)):
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X_train[tr]), sc.transform(X_train[va])
        m = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', n_jobs=-1)
        m.fit(Xtr, y[tr])
        acc = accuracy_score(y[va], m.predict(Xva))
        scores.append(acc)
        log(f"  Fold {fold+1}: {acc:.4f}")
    log(f"  Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # 全数据训练 + 预测
    log("\n生成测试集预测...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    final = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', n_jobs=-1)
    final.fit(X_train_s, y)
    test_preds = final.predict(X_test_s)
    
    predictions = {p: int(pred) for p, pred in zip(test_paths, test_preds)}
    os.makedirs(os.path.dirname(PRED_FILE), exist_ok=True)
    with open(PRED_FILE, 'w') as f:
        f.write(repr(predictions) + '\n')
    log(f"[✓] {PRED_FILE}")
    
    pred_dist = Counter(test_preds.tolist())
    log(f"\n预测分布: {dict(sorted(pred_dist.items()))}")
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(_log) + '\n')


if __name__ == "__main__":
    main()