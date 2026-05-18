# training/task2_train_v2.py
"""
Task 2 v2:
- 时间点切分 (melody + bass)
- 边界对比多粒度 (N=4, 8, 12)
- 整段特征改成 diff 形式
- 对称数据增强
- GBM
"""
import os
import numpy as np
import miditoolkit
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATAROOT = "student_files/task2_next_sequence_prediction"
PRED_FILE = "results/predictions2.json"
LOG_FILE = "train_log/task2_results_v2.txt"

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 1. 单段 MIDI 处理：提取时间点序列
# ============================================================
INTERVAL_MIN, INTERVAL_MAX = -12, 12
N_INTERVALS = INTERVAL_MAX - INTERVAL_MIN + 1


def parse_to_events(midi_path):
    """读 midi -> 按 start 分组的时间点序列
    返回:
      sorted_starts: list of int, 每个时间点的 start
      melody:        list of int, 每个时间点最高音 pitch
      bass:          list of int, 每个时间点最低音 pitch
      events_full:   defaultdict, start -> list of notes
    """
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    events = defaultdict(list)
    for n in notes:
        events[n.start].append(n)
    
    sorted_starts = sorted(events.keys())
    melody = [max(nn.pitch for nn in events[s]) for s in sorted_starts]
    bass = [min(nn.pitch for nn in events[s]) for s in sorted_starts]
    
    return sorted_starts, melody, bass, events


def segment_stats(starts, melody, bass, events):
    """整段统计：50 维"""
    if len(melody) < 2:
        return np.zeros(50, dtype=np.float32)
    
    pitches = np.array(melody + bass)
    
    # Chroma (12)
    chroma = np.zeros(12, dtype=np.float32)
    for p in pitches:
        chroma[p % 12] += 1
    chroma /= chroma.sum() + 1e-9
    
    # Melody interval hist (15 维, ±7)
    mi = np.zeros(15, dtype=np.float32)
    for iv in np.clip(np.diff(melody), -7, 7):
        mi[iv + 7] += 1
    mi /= mi.sum() + 1e-9
    
    # Bass interval hist (15 维, ±7)
    bi = np.zeros(15, dtype=np.float32)
    if len(bass) >= 2:
        for iv in np.clip(np.diff(bass), -7, 7):
            bi[iv + 7] += 1
        bi /= bi.sum() + 1e-9
    
    # 标量 (8)
    all_notes = [nn for s in starts for nn in events[s]]
    velocities = np.array([nn.velocity for nn in all_notes])
    durations = np.array([nn.end - nn.start for nn in all_notes])
    
    scalars = np.array([
        np.log1p(len(all_notes)),
        pitches.mean() / 127,
        pitches.std() / 24,
        velocities.mean() / 127,
        velocities.std() / 64,
        durations.mean() / 480,
        float(pitches.max() - pitches.min()) / 88,
        np.log1p(starts[-1] - starts[0] + 1) if len(starts) > 1 else 0,
    ], dtype=np.float32)
    
    return np.concatenate([chroma, mi, bi, scalars])  # 12+15+15+8 = 50


def boundary_features_v2(starts, melody, bass, events, side='end', n_tp=4):
    """边界 N 个时间点的特征
    side='end': 取最后 n_tp 个时间点
    side='start': 取最前 n_tp 个时间点
    返回 25 维
    """
    if len(starts) < 1:
        return np.zeros(25, dtype=np.float32)
    
    if side == 'end':
        sub_idx = list(range(max(0, len(starts) - n_tp), len(starts)))
    else:
        sub_idx = list(range(0, min(n_tp, len(starts))))
    
    sub_starts = [starts[i] for i in sub_idx]
    sub_melody = [melody[i] for i in sub_idx]
    sub_bass = [bass[i] for i in sub_idx]
    sub_notes = [nn for i in sub_idx for nn in events[starts[i]]]
    
    pitches = np.array(sub_melody + sub_bass)
    velocities = np.array([nn.velocity for nn in sub_notes])
    
    # Chroma (12)
    chroma = np.zeros(12, dtype=np.float32)
    for p in pitches:
        chroma[p % 12] += 1
    chroma /= chroma.sum() + 1e-9
    
    # 关键音符（最边缘那一个时间点）
    edge_idx = -1 if side == 'end' else 0
    edge_melody = sub_melody[edge_idx]
    edge_bass = sub_bass[edge_idx]
    
    # 边界处的 IOI 趋势
    if len(sub_starts) >= 2:
        iois = np.diff(sub_starts)
        ioi_mean = float(np.log1p(iois.mean()))
        ioi_last = float(np.log1p(iois[-1 if side == 'end' else 0]))
    else:
        ioi_mean = ioi_last = 0.0
    
    # 标量 (13)
    scalars = np.array([
        edge_melody / 127.0,         # 最边缘 melody
        edge_bass / 127.0,           # 最边缘 bass
        (edge_melody % 12) / 12.0,   # 最边缘 melody chroma
        (edge_bass % 12) / 12.0,
        pitches.mean() / 127,
        pitches.std() / 24 if len(pitches) > 1 else 0,
        float(pitches.max()) / 127,
        float(pitches.min()) / 127,
        velocities.mean() / 127 if len(velocities) else 0,
        ioi_mean,
        ioi_last,
        float(np.mean(sub_melody)) / 127,
        float(np.mean(sub_bass)) / 127,
    ], dtype=np.float32)
    
    return np.concatenate([chroma, scalars])  # 12 + 13 = 25


def pair_features_v2(starts_A, melody_A, bass_A, events_A,
                     starts_B, melody_B, bass_B, events_B):
    """对一对 (A, B) 提取特征
    设计：
    - 整段 diff: A 整段 - B 整段 (50), |A - B| (50)
    - 边界对比（多粒度 N=4, 8, 12）:
        每个 N 下: A_end vs B_start 的相似度, B_end vs A_start 的相似度, 差值
    """
    # 整段 stats
    stat_A = segment_stats(starts_A, melody_A, bass_A, events_A)
    stat_B = segment_stats(starts_B, melody_B, bass_B, events_B)
    
    diff = stat_A - stat_B
    abs_diff = np.abs(diff)
    
    # 多粒度边界
    boundary_feats = []
    for n_tp in [4, 8, 12]:
        A_end = boundary_features_v2(starts_A, melody_A, bass_A, events_A, 'end', n_tp)
        A_start = boundary_features_v2(starts_A, melody_A, bass_A, events_A, 'start', n_tp)
        B_end = boundary_features_v2(starts_B, melody_B, bass_B, events_B, 'end', n_tp)
        B_start = boundary_features_v2(starts_B, melody_B, bass_B, events_B, 'start', n_tp)
        
        # 两种假设下的边界差
        d_AB = A_end - B_start      # 假设 A→B
        d_BA = B_end - A_start      # 假设 B→A
        abs_d_AB = np.abs(d_AB)
        abs_d_BA = np.abs(d_BA)
        
        # 相似度标量
        sim_AB_l2 = -float(np.linalg.norm(d_AB))
        sim_BA_l2 = -float(np.linalg.norm(d_BA))
        sim_AB_l1 = -float(np.sum(abs_d_AB))
        sim_BA_l1 = -float(np.sum(abs_d_BA))
        
        # cosine（数值稳定的版本）
        ne_AB = np.linalg.norm(A_end) * np.linalg.norm(B_start) + 1e-9
        ne_BA = np.linalg.norm(B_end) * np.linalg.norm(A_start) + 1e-9
        cos_AB = float(np.dot(A_end, B_start) / ne_AB)
        cos_BA = float(np.dot(B_end, A_start) / ne_BA)
        
        # 边缘音符 pitch 差（最经典信号）
        last_m_A = melody_A[-1] if melody_A else 0
        first_m_A = melody_A[0] if melody_A else 0
        last_m_B = melody_B[-1] if melody_B else 0
        first_m_B = melody_B[0] if melody_B else 0
        
        pitch_gap_AB = abs(last_m_A - first_m_B) / 24.0   # A→B 边界
        pitch_gap_BA = abs(last_m_B - first_m_A) / 24.0   # B→A 边界
        
        # 类似的 bass 边界
        last_b_A = bass_A[-1] if bass_A else 0
        first_b_A = bass_A[0] if bass_A else 0
        last_b_B = bass_B[-1] if bass_B else 0
        first_b_B = bass_B[0] if bass_B else 0
        
        bass_gap_AB = abs(last_b_A - first_b_B) / 24.0
        bass_gap_BA = abs(last_b_B - first_b_A) / 24.0
        
        scalars = np.array([
            sim_AB_l2, sim_BA_l2, sim_AB_l2 - sim_BA_l2,
            sim_AB_l1, sim_BA_l1, sim_AB_l1 - sim_BA_l1,
            cos_AB, cos_BA, cos_AB - cos_BA,
            pitch_gap_AB, pitch_gap_BA, pitch_gap_AB - pitch_gap_BA,
            bass_gap_AB, bass_gap_BA, bass_gap_AB - bass_gap_BA,
        ], dtype=np.float32)
        
        boundary_feats.extend([d_AB, d_BA, abs_d_AB, abs_d_BA, scalars])
    
    # 最终拼接：
    # diff (50) + abs_diff (50) + boundary [N=4,8,12 各 (25+25+25+25+15)=115]
    # = 50 + 50 + 115*3 = 445
    return np.concatenate([diff, abs_diff] + boundary_feats).astype(np.float32)


# ============================================================
# 2. 数据加载
# ============================================================
def load_data():
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test = eval(f.read())
    return train, test


_event_cache = {}
def get_events(path):
    if path not in _event_cache:
        _event_cache[path] = parse_to_events(os.path.join(DATAROOT, path))
    return _event_cache[path]


def extract_features_for_pair(p1, p2):
    s_A, m_A, b_A, e_A = get_events(p1)
    s_B, m_B, b_B, e_B = get_events(p2)
    return pair_features_v2(s_A, m_A, b_A, e_A, s_B, m_B, b_B, e_B)


def extract_all_pairs(pairs, name="data", augment=False):
    """提取所有 pair 特征
    augment=True 时，每个 (A, B, label) 自动加 (B, A, ¬label)
    """
    feats = []
    labels = []
    pair_list = []  # 记录最终的 pair 顺序（测试集预测要用）
    
    for (p1, p2, lbl) in tqdm(pairs, desc=f"Extracting {name}"):
        # 原版
        feats.append(extract_features_for_pair(p1, p2))
        labels.append(lbl)
        pair_list.append((p1, p2))
        
        # 对称增强
        if augment:
            feats.append(extract_features_for_pair(p2, p1))
            labels.append(1 - lbl)
            pair_list.append((p2, p1))
    
    return np.array(feats, dtype=np.float32), np.array(labels), pair_list


# ============================================================
# 3. 训练 + 评估
# ============================================================
def cv_evaluate(X, y, model_fn, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        m = model_fn()
        m.fit(Xtr, y[tr])
        acc = accuracy_score(y[va], m.predict(Xva))
        scores.append(acc)
        log(f"  Fold {fold+1}: acc = {acc:.4f}")
    log(f"  Mean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores)


def main():
    log("=" * 70)
    log("Task 2 v2: 边界连贯性 + 多粒度 + 对称增强")
    log("=" * 70)
    
    train_meta, test_pairs = load_data()
    train_pairs_with_labels = [(p[0], p[1], 1 if v else 0) for p, v in train_meta.items()]
    
    log(f"训练对: {len(train_pairs_with_labels)}, 测试对: {len(test_pairs)}")
    
    # 训练特征（带对称增强）
    log("\n--- 训练特征（对称增强）---")
    X_train, y_train, _ = extract_all_pairs(train_pairs_with_labels, "train", augment=True)
    log(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    log(f"True 比例: {y_train.mean():.4f}")
    
    # 测试特征（不增强）
    log("\n--- 测试特征 ---")
    test_pairs_dummy = [(p[0], p[1], 0) for p in test_pairs]
    X_test, _, _ = extract_all_pairs(test_pairs_dummy, "test", augment=False)
    log(f"X_test: {X_test.shape}")
    log(f"缓存大小（不同 MIDI 文件数）: {len(_event_cache)}")
    
    # 三个模型对比
    log("\n" + "=" * 70)
    log("Logistic Regression")
    log("=" * 70)
    lr_acc = cv_evaluate(X_train, y_train,
                         lambda: LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1))
    
    log("\n" + "=" * 70)
    log("Random Forest")
    log("=" * 70)
    rf_acc = cv_evaluate(X_train, y_train,
                         lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                         random_state=42, n_jobs=-1))
    
    log("\n" + "=" * 70)
    log("Gradient Boosting")
    log("=" * 70)
    gbm_acc = cv_evaluate(X_train, y_train,
                          lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                              learning_rate=0.05, random_state=42))
    
    # 选最优
    cands = [
        ("LR", lr_acc, lambda: LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)),
        ("RF", rf_acc, lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                       random_state=42, n_jobs=-1)),
        ("GBM", gbm_acc, lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                                             learning_rate=0.05, random_state=42)),
    ]
    best_name, best_acc, make_best = max(cands, key=lambda x: x[1])
    log(f"\n>>> 最佳模型: {best_name} (CV acc = {best_acc:.4f})")
    
    # 全数据训练 + 测试预测
    log(f"\n在全训练集（含增强 {len(X_train)} 对）重训 {best_name}...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    final = make_best()
    final.fit(X_train_s, y_train)
    test_preds = final.predict(X_test_s)
    
    predictions = {pair: bool(p) for pair, p in zip(test_pairs, test_preds)}
    os.makedirs(os.path.dirname(PRED_FILE), exist_ok=True)
    with open(PRED_FILE, 'w') as f:
        f.write(repr(predictions) + '\n')
    log(f"\n[✓] {PRED_FILE}")
    log(f"预测数: {len(predictions)}")
    log(f"预测 True 比例: {sum(predictions.values()) / len(predictions):.4f}")
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(_log) + '\n')


if __name__ == "__main__":
    main()