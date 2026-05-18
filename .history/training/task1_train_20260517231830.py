# training/task1_train.py
"""
Task 1: Composer Classification - 正式训练脚本
特征: G 配置 (341 维)
  - Pitch histogram (88)
  - Chroma histogram (12)
  - Interval melody/bass (25+25, 不 mod 12)
  - Interval transition mod 12 (144, melody+bass 相加)
  - Duration histogram (12)
  - Velocity histogram (12)
  - IOI histogram (12)
  - Scalars (11，含 legato)
模型: LogisticRegression with class_weight='balanced'
"""
import os
import numpy as np
import miditoolkit
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ====== 路径 ======
DATAROOT = "student_files/task1_composer_classification"
PRED_FILE = "results/predictions1.json"
LOG_FILE = "train_log/task1_results.txt"
CACHE_FILE = "statistics/task1_features_G.npz"  # 复用之前的缓存
# ===================

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 特征提取
# ============================================================
PITCH_MIN, PITCH_MAX = 21, 108
N_PITCHES = PITCH_MAX - PITCH_MIN + 1
INTERVAL_MIN, INTERVAL_MAX = -12, 12
N_INTERVALS = INTERVAL_MAX - INTERVAL_MIN + 1
FEATURE_DIM = 341


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
    """一阶 interval histogram, 不 mod 12"""
    if len(pitch_sequence) < 2:
        return np.zeros(N_INTERVALS, dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ih = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX):
        ih[iv - INTERVAL_MIN] += 1
    ih /= ih.sum() + 1e-9
    return ih


def interval_transition_mod12(pitch_sequence):
    """二阶 interval transition, mod 12（计数矩阵，未归一化）"""
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


def extract_features(midi_path):
    """G 配置：341 维"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    if len(notes) < 2:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    # 按 start 分组得到时间点序列
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
    
    # interval melody/bass (25+25)
    im = interval_hist(melody_pitches)
    ib = interval_hist(bass_pitches)
    
    # interval transition mod 12, melody+bass 相加，行归一化 (144)
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
    
    feat = np.concatenate([ph, ch, im, ib, tr.flatten(), dh, vh, ih2, scalars])
    assert feat.shape[0] == FEATURE_DIM, f"特征维度错: {feat.shape[0]} != {FEATURE_DIM}"
    return feat


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 70)
    log("Task 1: Composer Classification - 正式训练")
    log("=" * 70)
    
    # 加载数据
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test_paths = eval(f.read())
    
    train_paths = list(train.keys())
    y = np.array([train[p] for p in train_paths])
    
    log(f"训练样本: {len(train_paths)}, 测试样本: {len(test_paths)}")
    log(f"composer 数: {len(set(y))}")
    
    # ---- 特征提取（训练 + 测试）----
    # 尝试用缓存
    use_cache = False
    if os.path.exists(CACHE_FILE):
        cache = np.load(CACHE_FILE, allow_pickle=True)
        cached_paths = cache['paths'].tolist()
        if cached_paths == train_paths:
            log(f"\n[✓] 复用训练集特征缓存: {CACHE_FILE}")
            X_train = cache['X']
            use_cache = True
    
    if not use_cache:
        log("\n--- 提取训练集特征 ---")
        X_train = np.array(
            [extract_features(os.path.join(DATAROOT, p)) for p in tqdm(train_paths)],
            dtype=np.float32
        )
    
    log("\n--- 提取测试集特征 ---")
    X_test = np.array(
        [extract_features(os.path.join(DATAROOT, p)) for p in tqdm(test_paths)],
        dtype=np.float32
    )
    log(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # ---- 5-fold CV 验证 ----
    log("\n" + "=" * 70)
    log("5-fold Stratified CV 验证")
    log("=" * 70)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    all_y_true, all_y_pred = [], []
    for fold, (tr, va) in enumerate(skf.split(X_train, y)):
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X_train[tr]), sc.transform(X_train[va])
        m = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', n_jobs=-1)
        m.fit(Xtr, y[tr])
        pred = m.predict(Xva)
        acc = accuracy_score(y[va], pred)
        cv_scores.append(acc)
        all_y_true.extend(y[va].tolist())
        all_y_pred.extend(pred.tolist())
        log(f"  Fold {fold+1}: acc = {acc:.4f}")
    log(f"  Mean ± Std: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # OOF confusion matrix
    log("\n--- OOF Confusion Matrix ---")
    cm = confusion_matrix(all_y_true, all_y_pred)
    log("       " + "  ".join(f"p{i:>2}" for i in range(8)))
    for i, row in enumerate(cm):
        log(f"  t{i:>2} | " + "  ".join(f"{v:>3d}" for v in row))
    
    # 每类准确率
    log("\n--- 每个 composer 的准确率 ---")
    y_arr = np.array(all_y_true)
    p_arr = np.array(all_y_pred)
    for cid in range(8):
        mask = (y_arr == cid)
        if mask.sum() > 0:
            acc = (p_arr[mask] == cid).mean()
            log(f"  composer {cid} ({mask.sum():>3} 样本): {acc:.4f}")
    
    # ---- 全训练集重训 → 测试集预测 ----
    log("\n" + "=" * 70)
    log("在全训练集上重训，生成测试集预测")
    log("=" * 70)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    final_model = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', n_jobs=-1)
    final_model.fit(X_train_s, y)
    test_preds = final_model.predict(X_test_s)
    
    # 保存预测
    predictions = {p: int(pred) for p, pred in zip(test_paths, test_preds)}
    os.makedirs(os.path.dirname(PRED_FILE), exist_ok=True)
    with open(PRED_FILE, 'w') as f:
        f.write(repr(predictions) + '\n')
    log(f"\n[✓] 预测已写入: {PRED_FILE}")
    log(f"预测数: {len(predictions)}")
    
    # 检查预测的分布
    pred_dist = Counter(test_preds.tolist())
    train_dist = Counter(y.tolist())
    log("\n--- 预测分布 vs 训练分布 ---")
    log(f"{'composer':>10} | {'train':>8} | {'train %':>8} | {'test pred':>10} | {'test %':>8}")
    log("-" * 60)
    for cid in sorted(train_dist.keys()):
        tr_cnt = train_dist[cid]
        tr_pct = tr_cnt / len(y) * 100
        te_cnt = pred_dist.get(cid, 0)
        te_pct = te_cnt / len(test_preds) * 100
        log(f"{cid:>10} | {tr_cnt:>8} | {tr_pct:>7.2f}% | {te_cnt:>10} | {te_pct:>7.2f}%")
    
    # 写日志
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(_log) + '\n')
    print(f"\n[✓] 日志已写入: {LOG_FILE}")


if __name__ == "__main__":
    main()