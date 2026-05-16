# task1_train.py
"""
Task 1: Composer classification
- 直方图特征 + Logistic Regression / Random Forest
- 5-fold stratified CV 评估
- 生成 predictions1.json
"""
import os
import numpy as np
import miditoolkit
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ====== 修改成你本地路径 ======
DATAROOT = "/home/howard/ml_assignment1/student_files/task1_composer_classification"
OUTPUT_FILE = "task1_results.txt"
PRED_FILE = "predictions1.json"
# ==============================

# 日志辅助
_log_lines = []
def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))

# ============================================================
# 1. 特征提取
# ============================================================

# MIDI pitch 范围：钢琴是 21~108，但我们看到数据中是 24~105，统一用 21~108
PITCH_MIN, PITCH_MAX = 21, 108  # 88 个音
N_PITCHES = PITCH_MAX - PITCH_MIN + 1

# 音程范围：限制在 ±12 半音（一个八度），超出归到边界
INTERVAL_MIN, INTERVAL_MAX = -12, 12
N_INTERVALS = INTERVAL_MAX - INTERVAL_MIN + 1  # 25

# Duration / IOI / Velocity 用 log-binned 直方图
N_DUR_BINS = 12
N_IOI_BINS = 12
N_VEL_BINS = 12


def extract_features(midi_path):
    """从单个 MIDI 文件提取特征向量"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    
    # 合并所有 instruments 的 notes（数据里都是 1 个 instrument，但写得通用一些）
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    if len(notes) < 2:
        # 防御性：如果某个文件太短，返回零向量
        return np.zeros(get_feature_dim(), dtype=np.float32)
    
    # 按 start 排序（重要：interval 需要时间顺序）
    notes.sort(key=lambda n: (n.start, n.pitch))
    
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    # ---------- 1. Pitch histogram（88 维）----------
    pitch_hist = np.zeros(N_PITCHES, dtype=np.float32)
    clipped = np.clip(pitches, PITCH_MIN, PITCH_MAX)
    for p in clipped:
        pitch_hist[p - PITCH_MIN] += 1
    pitch_hist /= pitch_hist.sum() + 1e-9  # 归一化为比例
    
    # ---------- 2. Chroma histogram / Pitch class histogram（12 维）----------
    chroma_hist = np.zeros(12, dtype=np.float32)
    for p in pitches:
        chroma_hist[p % 12] += 1
    chroma_hist /= chroma_hist.sum() + 1e-9
    
    # ---------- 3. Interval histogram（25 维）----------
    intervals = np.diff(pitches)
    intervals_clipped = np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX)
    interval_hist = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in intervals_clipped:
        interval_hist[iv - INTERVAL_MIN] += 1
    interval_hist /= interval_hist.sum() + 1e-9
    
    # ---------- 4. Pitch class transition matrix（12*12 = 144 维）----------
    pc = pitches % 12
    transition = np.zeros((12, 12), dtype=np.float32)
    for i in range(len(pc) - 1):
        transition[pc[i], pc[i+1]] += 1
    # 行归一化（条件概率）
    row_sums = transition.sum(axis=1, keepdims=True)
    transition = transition / (row_sums + 1e-9)
    transition_flat = transition.flatten()
    
    # ---------- 5. Duration histogram (log-binned, 12 维) ----------
    # 用 log2(duration) 分箱
    dur_log = np.log2(np.maximum(durations, 1))
    dur_hist, _ = np.histogram(dur_log, bins=N_DUR_BINS, range=(0, 12))
    dur_hist = dur_hist.astype(np.float32)
    dur_hist /= dur_hist.sum() + 1e-9
    
    # ---------- 6. Velocity histogram（12 维）----------
    vel_hist, _ = np.histogram(velocities, bins=N_VEL_BINS, range=(0, 128))
    vel_hist = vel_hist.astype(np.float32)
    vel_hist /= vel_hist.sum() + 1e-9
    
    # ---------- 7. IOI (Inter-Onset Interval) histogram（12 维）----------
    iois = np.diff(starts)
    iois = iois[iois > 0]  # 去掉同时发声
    if len(iois) > 0:
        ioi_log = np.log2(np.maximum(iois, 1))
        ioi_hist, _ = np.histogram(ioi_log, bins=N_IOI_BINS, range=(0, 14))
        ioi_hist = ioi_hist.astype(np.float32)
        ioi_hist /= ioi_hist.sum() + 1e-9
    else:
        ioi_hist = np.zeros(N_IOI_BINS, dtype=np.float32)
    
    # ---------- 8. 标量特征（少量，归一化过的）----------
    # 这些是辅助信息，不是主力
    total_ticks = starts.max() - starts.min() + 1
    note_density = len(notes) / (total_ticks / 480.0)  # 每拍多少音符
    pitch_range = float(pitches.max() - pitches.min())
    pitch_std = float(pitches.std())
    # 多音性（polyphony）：同时发声的最大音符数（简化估计）
    poly = compute_polyphony(notes)
    
    scalars = np.array([
        np.log1p(len(notes)),
        np.log1p(total_ticks / 480.0),  # log of beat count
        note_density,
        pitch_range / 88.0,             # 归一化到 0~1
        pitch_std / 24.0,               # 标准差归一化
        float(pitches.mean()) / 127.0,  # 平均音高归一化
        float(velocities.mean()) / 127.0,
        float(velocities.std()) / 64.0,
        poly / 10.0,                    # 归一化
    ], dtype=np.float32)
    
    # 拼接
    feat = np.concatenate([
        pitch_hist,         # 88
        chroma_hist,        # 12
        interval_hist,      # 25
        transition_flat,    # 144
        dur_hist,           # 12
        vel_hist,           # 12
        ioi_hist,           # 12
        scalars,            # 9
    ])
    return feat


def compute_polyphony(notes):
    """估算多音性：扫描所有事件点，找最大同时发声数"""
    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    max_poly = 0
    cur = 0
    for _, delta in events:
        cur += delta
        max_poly = max(max_poly, cur)
    return max_poly


def get_feature_dim():
    return N_PITCHES + 12 + N_INTERVALS + 144 + N_DUR_BINS + N_VEL_BINS + N_IOI_BINS + 9


FEATURE_NAMES = (
    [f'pitch_{i+PITCH_MIN}' for i in range(N_PITCHES)] +
    [f'chroma_{i}' for i in range(12)] +
    [f'interval_{i+INTERVAL_MIN:+d}' for i in range(N_INTERVALS)] +
    [f'trans_{i}_{j}' for i in range(12) for j in range(12)] +
    [f'dur_bin_{i}' for i in range(N_DUR_BINS)] +
    [f'vel_bin_{i}' for i in range(N_VEL_BINS)] +
    [f'ioi_bin_{i}' for i in range(N_IOI_BINS)] +
    ['log_n_notes', 'log_n_beats', 'note_density', 'pitch_range',
     'pitch_std', 'pitch_mean', 'vel_mean', 'vel_std', 'polyphony']
)

# ============================================================
# 2. 读取数据 + 提取特征
# ============================================================

def load_data():
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    with open(os.path.join(DATAROOT, "test.json")) as f:
        test = eval(f.read())
    return train, test


def extract_all(paths, name="data"):
    feats = []
    for p in tqdm(paths, desc=f"Extracting {name}"):
        f = extract_features(os.path.join(DATAROOT, p))
        feats.append(f)
    return np.array(feats)


# ============================================================
# 3. 评估 + 训练
# ============================================================

def cv_evaluate(X, y, model_fn, n_splits=5, seed=42):
    """5-fold stratified CV，返回平均准确率"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    all_y_true, all_y_pred = [], []
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        
        model = model_fn()
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_va_s)
        
        acc = accuracy_score(y_va, pred)
        scores.append(acc)
        all_y_true.extend(y_va.tolist())
        all_y_pred.extend(pred.tolist())
        log(f"  Fold {fold+1}: acc = {acc:.4f}")
    
    log(f"  Mean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores), all_y_true, all_y_pred


def main():
    log("=" * 70)
    log("Task 1: Composer Classification - 训练 pipeline")
    log("=" * 70)
    
    train_meta, test_paths = load_data()
    log(f"训练样本: {len(train_meta)}, 测试样本: {len(test_paths)}")
    log(f"特征维度: {get_feature_dim()}")
    
    # 提取特征
    train_paths_list = list(train_meta.keys())
    y = np.array([train_meta[p] for p in train_paths_list])
    
    log("\n--- 特征提取 ---")
    X_train = extract_all(train_paths_list, "train")
    X_test = extract_all(test_paths, "test")
    log(f"X_train shape: {X_train.shape}")
    log(f"X_test shape: {X_test.shape}")
    
    # CV 评估 - Logistic Regression
    log("\n" + "=" * 70)
    log("Logistic Regression (class_weight='balanced')")
    log("=" * 70)
    def make_lr():
        return LogisticRegression(
            max_iter=3000, C=1.0, class_weight='balanced',
            solver='lbfgs', multi_class='auto', n_jobs=-1
        )
    lr_acc, _, _ = cv_evaluate(X_train, y, make_lr)
    
    # CV 评估 - Random Forest
    log("\n" + "=" * 70)
    log("Random Forest (class_weight='balanced')")
    log("=" * 70)
    def make_rf():
        return RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    rf_acc, rf_y_true, rf_y_pred = cv_evaluate(X_train, y, make_rf)
    
    # 选最好的模型
    if rf_acc >= lr_acc:
        best_name = "RandomForest"
        best_acc = rf_acc
        make_best = make_rf
    else:
        best_name = "LogisticRegression"
        best_acc = lr_acc
        make_best = make_lr
    
    log(f"\n>>> 最佳模型: {best_name} (CV acc = {best_acc:.4f})")
    
    # 在 RF 的 CV 预测上打印 confusion matrix（看看哪类错得多）
    log(f"\n--- {best_name} CV confusion matrix（行：真，列：预测）---")
    cm = confusion_matrix(rf_y_true if best_name == "RandomForest" else _,
                          rf_y_pred if best_name == "RandomForest" else _)
    if best_name == "RandomForest":
        log("       " + "  ".join(f"p{i:>2}" for i in range(8)))
        for i, row in enumerate(cm):
            log(f"  t{i:>2} | " + "  ".join(f"{v:>3d}" for v in row))
        log("\n--- per-class report ---")
        log(classification_report(rf_y_true, rf_y_pred, zero_division=0))
    
    # 全训练集重训，生成测试集预测
    log("\n" + "=" * 70)
    log("在全训练集上重训，生成 predictions1.json")
    log("=" * 70)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    final_model = make_best()
    final_model.fit(X_train_s, y)
    test_preds = final_model.predict(X_test_s)
    
    # 保存预测（autograder 用 eval 读取，且 key 顺序要保持）
    predictions = {p: int(pred) for p, pred in zip(test_paths, test_preds)}
    
    # 用 repr() 保存（和 baseline 的 write_submission_predictions 一致）
    with open(PRED_FILE, 'w') as f:
        f.write(repr(predictions) + '\n')
    log(f"[✓] 预测已写入: {PRED_FILE}")
    log(f"预测数: {len(predictions)}")
    
    # 预测的标签分布
    pred_dist = Counter(test_preds.tolist())
    log(f"\n预测的 composer 分布:")
    for cid in sorted(pred_dist.keys()):
        log(f"  composer {cid}: {pred_dist[cid]}")
    
    # 写日志
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(_log_lines) + '\n')
    print(f"\n[✓] 日志已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()