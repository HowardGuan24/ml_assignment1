# training/task1_train_v3.py
"""
Task 1 训练 v3:
- 保留 v2 的所有特征 (G 配置 + 基础 metadata)
- 新增：velocity-128 binary（128 维），tempo TOP-30 one-hot，duration TOP-15 one-hot
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
LOG_FILE = "train_log/task1_results_v3.txt"

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

ALL_KEYS = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb',
            'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
            'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bbm', 'Bm']
N_KEYS = len(ALL_KEYS)
KEY_TO_IDX = {k: i for i, k in enumerate(ALL_KEYS)}

# 这些会在 main() 里根据训练集动态生成
TEMPO_VOCAB = None       # TOP-K 整数 tempo 值
DURATION_VOCAB = None    # TOP-K most_common_duration 值
TIMESIG_VOCAB = None     # (num, den) 组合


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


def build_vocab(train_paths, top_k_tempo=30, top_k_dur=15):
    """扫描训练集生成 tempo / duration / timesig 词表"""
    log("--- 构建词表（扫描训练集）---")
    tempo_counter = Counter()
    dur_counter = Counter()
    timesig_counter = Counter()
    
    for p in tqdm(train_paths, desc="vocab scan"):
        midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, p))
        # tempo: 取整数部分
        if midi.tempo_changes:
            t = int(round(midi.tempo_changes[0].tempo))
            tempo_counter[t] += 1
        # most_common_duration
        notes = []
        for inst in midi.instruments:
            notes.extend(inst.notes)
        if notes:
            durs = [n.end - n.start for n in notes]
            most_dur = Counter(durs).most_common(1)[0][0]
            dur_counter[most_dur] += 1
        # time sig
        if midi.time_signature_changes:
            ts = midi.time_signature_changes[0]
            timesig_counter[(ts.numerator, ts.denominator)] += 1
    
    tempo_vocab = [t for t, _ in tempo_counter.most_common(top_k_tempo)]
    dur_vocab = [d for d, _ in dur_counter.most_common(top_k_dur)]
    timesig_vocab = [ts for ts, _ in timesig_counter.most_common(15)]
    
    log(f"Tempo TOP {top_k_tempo}: {tempo_vocab[:10]}...")
    log(f"Duration TOP {top_k_dur}: {dur_vocab}")
    log(f"TimeSig TOP {len(timesig_vocab)}: {timesig_vocab}")
    
    return tempo_vocab, dur_vocab, timesig_vocab


def extract_features(midi_path):
    """G 配置 + 完整 metadata + 强指纹特征"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    expected_dim = (N_PITCHES + 12 + N_INTERVALS*2 + 144 + 12 + 12 + 12 + 11
                    + 8 + N_KEYS  # 基础 metadata
                    + 128         # velocity-128 binary
                    + len(TEMPO_VOCAB) + 1
                    + len(DURATION_VOCAB) + 1
                    + len(TIMESIG_VOCAB) + 1)
    
    if len(notes) < 2:
        return np.zeros(expected_dim, dtype=np.float32)
    
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
    
    # ----- G 配置部分（341 维）-----
    ph = np.zeros(N_PITCHES, dtype=np.float32)
    for p in np.clip(pitches, PITCH_MIN, PITCH_MAX):
        ph[p - PITCH_MIN] += 1
    ph /= ph.sum() + 1e-9
    
    ch = np.zeros(12, dtype=np.float32)
    for p in pitches:
        ch[p % 12] += 1
    ch /= ch.sum() + 1e-9
    
    im = interval_hist(melody_pitches)
    ib = interval_hist(bass_pitches)
    
    tr_m = interval_transition_mod12(melody_pitches)
    tr_b = interval_transition_mod12(bass_pitches)
    tr = tr_m + tr_b
    tr = tr / (tr.sum(axis=1, keepdims=True) + 1e-9)
    
    dl = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dl, bins=12, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    
    vh, _ = np.histogram(velocities, bins=12, range=(0, 128))
    vh = vh.astype(np.float32) / (vh.sum() + 1e-9)
    
    ih2 = ioi_hist(sorted_starts)
    
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
    
    # ----- 基础 metadata（8 + N_KEYS 维）-----
    if midi.tempo_changes:
        first_tempo = midi.tempo_changes[0].tempo
    else:
        first_tempo = 120.0
    
    meta_basic = [
        first_tempo / 200.0,
        1.0 if abs(first_tempo - round(first_tempo)) < 1e-6 else 0.0,
        first_tempo - int(first_tempo),
        min(len(midi.tempo_changes), 10) / 10.0,
        (midi.time_signature_changes[0].numerator / 12.0) if midi.time_signature_changes else 4/12.0,
        (midi.time_signature_changes[0].denominator / 16.0) if midi.time_signature_changes else 4/16.0,
        min(len(midi.time_signature_changes), 5) / 5.0,
        min(len(midi.key_signature_changes), 5) / 5.0,
    ]
    key_oh = [0.0] * N_KEYS
    if midi.key_signature_changes:
        k = midi.key_signature_changes[0].key_name
        if k in KEY_TO_IDX:
            key_oh[KEY_TO_IDX[k]] = 1.0
    meta_basic = np.array(meta_basic + key_oh, dtype=np.float32)
    
    # ----- Velocity-128 binary（128 维）-----
    vel_set = set(velocities.tolist())
    vel_binary = np.zeros(128, dtype=np.float32)
    for v in vel_set:
        if 0 <= v < 128:
            vel_binary[v] = 1.0
    
    # ----- Tempo TOP-K one-hot（K+1 维，最后一位是 "other"）-----
    tempo_oh = np.zeros(len(TEMPO_VOCAB) + 1, dtype=np.float32)
    tempo_int = int(round(first_tempo))
    if tempo_int in TEMPO_VOCAB:
        tempo_oh[TEMPO_VOCAB.index(tempo_int)] = 1.0
    else:
        tempo_oh[-1] = 1.0  # other
    
    # ----- Most common duration TOP-K one-hot（K+1 维）-----
    most_dur = Counter(durations.tolist()).most_common(1)[0][0]
    dur_oh = np.zeros(len(DURATION_VOCAB) + 1, dtype=np.float32)
    if most_dur in DURATION_VOCAB:
        dur_oh[DURATION_VOCAB.index(most_dur)] = 1.0
    else:
        dur_oh[-1] = 1.0
    
    # ----- TimeSig TOP-K one-hot（K+1 维）-----
    ts_oh = np.zeros(len(TIMESIG_VOCAB) + 1, dtype=np.float32)
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        key = (ts.numerator, ts.denominator)
        if key in TIMESIG_VOCAB:
            ts_oh[TIMESIG_VOCAB.index(key)] = 1.0
        else:
            ts_oh[-1] = 1.0
    else:
        ts_oh[-1] = 1.0
    
    return np.concatenate([g_feat, meta_basic, vel_binary, tempo_oh, dur_oh, ts_oh])


def main():
    global TEMPO_VOCAB, DURATION_VOCAB, TIMESIG_VOCAB
    
    log("=" * 70)
    log("Task 1 v3: G 特征 + 完整 metadata + 强指纹")
    log("=" * 70)
    
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test_paths = eval(f.read())
    
    train_paths = list(train.keys())
    y = np.array([train[p] for p in train_paths])
    
    log(f"训练: {len(train_paths)}, 测试: {len(test_paths)}")
    
    # 构建词表（只用训练集，避免 leak）
    TEMPO_VOCAB, DURATION_VOCAB, TIMESIG_VOCAB = build_vocab(train_paths)
    
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