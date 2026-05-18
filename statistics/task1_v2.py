# statistics/task1_feature_v2.py
"""
Task 1 特征 v2：
- Interval：melody + bass，不 mod 12
- Transition：interval mod 12 后做 12x12，melody + bass 相加
- 加 legato ratio (mean, std)
- 对比 G (留 pitch) vs H (砍 pitch)
"""
import os
import numpy as np
import miditoolkit
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATAROOT = "student_files/task1_composer_classification"
OUTPUT_FILE = "statistics/task1_feature_v2.txt"

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
    """一阶 interval 直方图（不 mod 12，保留方向）"""
    if len(pitch_sequence) < 2:
        return np.zeros(N_INTERVALS, dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ih = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX):
        ih[iv - INTERVAL_MIN] += 1
    ih /= ih.sum() + 1e-9
    return ih


def interval_transition_mod12(pitch_sequence):
    """二阶 interval transition: 把 interval mod 12，然后做 12x12 转移
    返回未归一化的 12x12 矩阵（计数）"""
    if len(pitch_sequence) < 3:
        return np.zeros((12, 12), dtype=np.float32)
    intervals = np.diff(np.array(pitch_sequence))
    ic = intervals % 12   # interval class: 0~11
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


# ============================================================
# 特征提取
# ============================================================
def extract_features(midi_path):
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    groups = {}
    if len(notes) < 2:
        groups['pitch']            = np.zeros(N_PITCHES, dtype=np.float32)
        groups['chroma']           = np.zeros(12, dtype=np.float32)
        groups['interval_melody']  = np.zeros(N_INTERVALS, dtype=np.float32)
        groups['interval_bass']    = np.zeros(N_INTERVALS, dtype=np.float32)
        groups['transition_int12'] = np.zeros(144, dtype=np.float32)
        groups['duration']         = np.zeros(12, dtype=np.float32)
        groups['velocity_hist']    = np.zeros(12, dtype=np.float32)
        groups['ioi_new']          = np.zeros(12, dtype=np.float32)
        groups['scalars']          = np.zeros(11, dtype=np.float32)  # 9 + 2 legato
        return groups
    
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    # 按 start 分组，得到时间点序列
    events = defaultdict(list)
    for n in notes:
        events[n.start].append(n)
    sorted_starts = sorted(events.keys())
    melody_pitches = [max(nn.pitch for nn in events[s]) for s in sorted_starts]
    bass_pitches = [min(nn.pitch for nn in events[s]) for s in sorted_starts]
    
    # ---- Pitch (88) ----
    ph = np.zeros(N_PITCHES, dtype=np.float32)
    for p in np.clip(pitches, PITCH_MIN, PITCH_MAX):
        ph[p - PITCH_MIN] += 1
    ph /= ph.sum() + 1e-9
    groups['pitch'] = ph
    
    # ---- Chroma (12) ----
    ch = np.zeros(12, dtype=np.float32)
    for p in pitches:
        ch[p % 12] += 1
    ch /= ch.sum() + 1e-9
    groups['chroma'] = ch
    
    # ---- Interval melody / bass (25 + 25) ----
    groups['interval_melody'] = interval_hist(melody_pitches)
    groups['interval_bass']   = interval_hist(bass_pitches)
    
    # ---- Interval transition mod 12 (12x12 = 144) ----
    # melody transition + bass transition 相加，然后行归一化
    tr_m = interval_transition_mod12(melody_pitches)
    tr_b = interval_transition_mod12(bass_pitches)
    tr = tr_m + tr_b
    row_sums = tr.sum(axis=1, keepdims=True)
    tr = tr / (row_sums + 1e-9)
    groups['transition_int12'] = tr.flatten()
    
    # ---- Duration (12) ----
    dl = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dl, bins=12, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    groups['duration'] = dh
    
    # ---- Velocity hist (12) ----
    vh, _ = np.histogram(velocities, bins=12, range=(0, 128))
    vh = vh.astype(np.float32) / (vh.sum() + 1e-9)
    groups['velocity_hist'] = vh
    
    # ---- IOI new (12) ----
    groups['ioi_new'] = ioi_hist(sorted_starts)
    
    # ---- Legato ratio (每个时间点 longest_note_duration / next_onset_gap) ----
    legato_ratios = []
    for i in range(len(sorted_starts) - 1):
        s_cur = sorted_starts[i]
        s_next = sorted_starts[i + 1]
        gap = s_next - s_cur
        if gap > 0:
            # 该时间点最长的音符 duration
            max_dur = max(nn.end - nn.start for nn in events[s_cur])
            ratio = max_dur / gap
            legato_ratios.append(min(ratio, 2.0))  # clip 极端值
    
    if legato_ratios:
        legato_mean = float(np.mean(legato_ratios))
        legato_std = float(np.std(legato_ratios))
    else:
        legato_mean = 0.0
        legato_std = 0.0
    
    # ---- Scalars (9 + 2 legato = 11) ----
    total_ticks = max(starts.max() - starts.min(), 1)
    groups['scalars'] = np.array([
        np.log1p(len(notes)), np.log1p(total_ticks / 480.0),
        len(notes) / (total_ticks / 480.0),
        float(pitches.max() - pitches.min()) / 88,
        float(pitches.std()) / 24,
        float(pitches.mean()) / 127,
        float(velocities.mean()) / 127,
        float(velocities.std()) / 64,
        compute_polyphony(notes) / 10.0,
        legato_mean,
        legato_std,
    ], dtype=np.float32)
    
    return groups


# ============================================================
# 评估
# ============================================================
def eval_config(X, y, ids, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = []
    for tr, va in skf.split(X, y):
        sc = StandardScaler()
        Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
        m = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
        m.fit(Xtr, y[tr])
        cv_scores.append(accuracy_score(y[va], m.predict(Xva)))
    cv_mean = np.mean(cv_scores)
    
    sort_order = np.argsort(ids)
    n_train = int(len(ids) * 0.8)
    tr_idx = sort_order[:n_train]
    va_idx = sort_order[n_train:]
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr_idx]), sc.transform(X[va_idx])
    m = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
    m.fit(Xtr, y[tr_idx])
    ho_acc = accuracy_score(y[va_idx], m.predict(Xva))
    
    return cv_mean, ho_acc


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 75)
    log("Task 1 特征 v2 实验")
    log("=" * 75)
    
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    paths = list(train.keys())
    y = np.array([train[p] for p in paths])
    ids = np.array([int(p.split('/')[1].split('.')[0]) for p in paths])
    
    log(f"训练样本: {len(paths)}")
    
    # 提取所有特征组
    log("\n--- 特征提取 ---")
    all_groups = defaultdict(list)
    for p in tqdm(paths):
        groups = extract_features(os.path.join(DATAROOT, p))
        for name, vec in groups.items():
            all_groups[name].append(vec)
    
    for name in all_groups:
        all_groups[name] = np.array(all_groups[name], dtype=np.float32)
    
    log(f"\n各特征组维度:")
    for name, arr in all_groups.items():
        log(f"  {name:25s}: {arr.shape[1]:>4d}")
    
    # 对照实验
    configs = [
        ("B (旧基准: 含 pitch + PC transition + vel-hist)", None),  # 用上次结果，仅参考
        ("G. v2 + pitch",
            ['pitch', 'chroma', 'interval_melody', 'interval_bass',
             'transition_int12', 'duration', 'velocity_hist', 'ioi_new', 'scalars']),
        ("H. v2 - pitch",
            ['chroma', 'interval_melody', 'interval_bass',
             'transition_int12', 'duration', 'velocity_hist', 'ioi_new', 'scalars']),
    ]
    
    log("\n" + "=" * 75)
    log("跑对照实验")
    log("=" * 75)
    log(f"\n{'配置':<45} | {'维度':>5} | {'5-fold CV':>10} | {'ID hold-out':>12}")
    log("-" * 80)
    
    log(f"{'B (上次实验参考)':<45} | {339:>5} | {0.7843:>10.4f} | {0.7934:>12.4f}")
    
    results = []
    for name, group_names in configs[1:]:  # 跳过 B（直接用上次结果）
        X = np.concatenate([all_groups[n] for n in group_names], axis=1)
        cv, ho = eval_config(X, y, ids)
        log(f"{name:<45} | {X.shape[1]:>5} | {cv:>10.4f} | {ho:>12.4f}")
        results.append((name, X.shape[1], cv, ho))
    
    # 判定
    log("\n" + "=" * 75)
    log("判定")
    log("=" * 75)
    
    G_cv, G_ho = results[0][2], results[0][3]
    H_cv, H_ho = results[1][2], results[1][3]
    
    log(f"\nG (含 pitch): CV={G_cv:.4f}, hold-out={G_ho:.4f}")
    log(f"H (砍 pitch): CV={H_cv:.4f}, hold-out={H_ho:.4f}")
    log(f"差距：CV {H_cv - G_cv:+.4f}, hold-out {H_ho - G_ho:+.4f}")
    
    if H_ho >= G_ho - 0.005:
        log(">>> 砍 pitch 几乎无损失，建议用 H（更精简）")
    else:
        log(">>> 砍 pitch 损失明显，建议保留 pitch（用 G）")
    
    # 也对比新 transition 是否比旧 PC transition 强
    log(f"\n对比上次 B 的 (CV 0.7843, hold-out 0.7934):")
    log(f"  G 比 B: CV {G_cv - 0.7843:+.4f}, hold-out {G_ho - 0.7934:+.4f}")
    log(f"  H 比 B: CV {H_cv - 0.7843:+.4f}, hold-out {H_ho - 0.7934:+.4f}")
    log("  >>> 看 transition_int12 + legato 这两个新设计的总效果")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(_log) + '\n')
    log(f"\n[✓] 日志已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()