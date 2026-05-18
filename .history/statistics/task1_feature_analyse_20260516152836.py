# statistics/task1_feature_analysis.py
"""
Task 1 特征审视：
1. 每个特征组的"区分度"（不同 composer 的特征分布差异）
2. 每个特征组单独跑 LR 的 CV 准确率
3. velocity 是不是常数（专项检查）
4. 各特征组的方差（接近 0 = 没信息）
"""
import os
import sys
import numpy as np
import miditoolkit
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score

# ====== 路径（相对项目根目录运行）======
DATAROOT = "student_files/task1_composer_classification"
OUTPUT_FILE = "statistics/task1_feature_analysis.txt"
# =======================================

_log = []
def log(m=""):
    print(m); _log.append(str(m))


# ============================================================
# 特征提取：返回 dict，按特征组分开，方便后面分析
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


def extract_feature_groups(midi_path):
    """返回 dict: {特征组名: ndarray}，便于单独分析每组"""
    midi = miditoolkit.midi.parser.MidiFile(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    
    groups = {}
    if len(notes) < 2:
        # 防御：太短就全零
        groups['pitch']      = np.zeros(N_PITCHES, dtype=np.float32)
        groups['chroma']     = np.zeros(12, dtype=np.float32)
        groups['interval']   = np.zeros(N_INTERVALS, dtype=np.float32)
        groups['transition'] = np.zeros(144, dtype=np.float32)
        groups['duration']   = np.zeros(12, dtype=np.float32)
        groups['velocity']   = np.zeros(12, dtype=np.float32)
        groups['ioi']        = np.zeros(12, dtype=np.float32)
        groups['scalars']    = np.zeros(9, dtype=np.float32)
        return groups
    
    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    durations = np.array([n.end - n.start for n in notes])
    starts = np.array([n.start for n in notes])
    
    # 1. Pitch histogram (88) - 怀疑组
    ph = np.zeros(N_PITCHES, dtype=np.float32)
    for p in np.clip(pitches, PITCH_MIN, PITCH_MAX):
        ph[p - PITCH_MIN] += 1
    ph /= ph.sum() + 1e-9
    groups['pitch'] = ph
    
    # 2. Chroma histogram (12) - 教授强推
    ch = np.zeros(12, dtype=np.float32)
    for p in pitches:
        ch[p % 12] += 1
    ch /= ch.sum() + 1e-9
    groups['chroma'] = ch
    
    # 3. Interval histogram (25) - 教授强推
    intervals = np.diff(pitches)
    ih = np.zeros(N_INTERVALS, dtype=np.float32)
    for iv in np.clip(intervals, INTERVAL_MIN, INTERVAL_MAX):
        ih[iv - INTERVAL_MIN] += 1
    ih /= ih.sum() + 1e-9
    groups['interval'] = ih
    
    # 4. PC transition (144) - 怀疑过拟合
    pc = pitches % 12
    tr = np.zeros((12, 12), dtype=np.float32)
    for i in range(len(pc) - 1):
        tr[pc[i], pc[i+1]] += 1
    tr = tr / (tr.sum(axis=1, keepdims=True) + 1e-9)
    groups['transition'] = tr.flatten()
    
    # 5. Duration histogram (12)
    dl = np.log2(np.maximum(durations, 1))
    dh, _ = np.histogram(dl, bins=12, range=(0, 12))
    dh = dh.astype(np.float32) / (dh.sum() + 1e-9)
    groups['duration'] = dh
    
    # 6. Velocity histogram (12) - 怀疑是常数
    vh, _ = np.histogram(velocities, bins=12, range=(0, 128))
    vh = vh.astype(np.float32) / (vh.sum() + 1e-9)
    groups['velocity'] = vh
    
    # 7. IOI histogram (12)
    iois = np.diff(starts); iois = iois[iois > 0]
    if len(iois) > 0:
        il = np.log2(np.maximum(iois, 1))
        ioi, _ = np.histogram(il, bins=12, range=(0, 14))
        ioi = ioi.astype(np.float32) / (ioi.sum() + 1e-9)
    else:
        ioi = np.zeros(12, dtype=np.float32)
    groups['ioi'] = ioi
    
    # 8. Scalars (9)
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
    ], dtype=np.float32)
    
    return groups


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 70)
    log("Task 1 特征审视")
    log("=" * 70)
    
    # 读数据
    with open(os.path.join(DATAROOT, "train.json")) as f:
        train = eval(f.read())
    paths = list(train.keys())
    y = np.array([train[p] for p in paths])
    
    log(f"训练样本: {len(paths)}, composer 数: {len(set(y))}")
    
    # 提取所有特征组
    log("\n--- 特征提取 ---")
    all_groups = {name: [] for name in
                  ['pitch', 'chroma', 'interval', 'transition',
                   'duration', 'velocity', 'ioi', 'scalars']}
    
    for p in tqdm(paths):
        groups = extract_feature_groups(os.path.join(DATAROOT, p))
        for name, vec in groups.items():
            all_groups[name].append(vec)
    
    for name in all_groups:
        all_groups[name] = np.array(all_groups[name], dtype=np.float32)
    
    log(f"\n各特征组维度:")
    for name, arr in all_groups.items():
        log(f"  {name:12s}: {arr.shape}")
    
    # ========================================================
    # 分析 1：velocity 真的是常数吗？做专项检查
    # ========================================================
    log("\n" + "=" * 70)
    log("分析 1: Velocity 是不是常数？")
    log("=" * 70)
    
    # 在原始 velocity 值上统计（不是分箱后）
    log("\n检查所有训练文件的 velocity 取值分布...")
    all_velocities = []
    velocity_per_file_std = []
    for p in tqdm(paths[:300]):  # 采样 300 个
        midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, p))
        notes = []
        for inst in midi.instruments:
            if not inst.is_drum:
                notes.extend(inst.notes)
        if not notes:
            continue
        vels = [n.velocity for n in notes]
        all_velocities.extend(vels)
        velocity_per_file_std.append(np.std(vels))
    
    all_velocities = np.array(all_velocities)
    velocity_per_file_std = np.array(velocity_per_file_std)
    
    log(f"汇总 {len(all_velocities)} 个音符的 velocity:")
    log(f"  唯一值数: {len(np.unique(all_velocities))}")
    log(f"  取值范围: {all_velocities.min()} - {all_velocities.max()}")
    log(f"  均值: {all_velocities.mean():.2f}, 标准差: {all_velocities.std():.2f}")
    log(f"  TOP 10 取值（按频率）:")
    vel_cnt = Counter(all_velocities.tolist())
    for v, c in vel_cnt.most_common(10):
        log(f"    velocity={v}: {c} 次 ({c/len(all_velocities)*100:.1f}%)")
    
    log(f"\n每个文件内 velocity 的标准差（前 300 个文件）:")
    log(f"  均值: {velocity_per_file_std.mean():.2f}")
    log(f"  中位数: {np.median(velocity_per_file_std):.2f}")
    log(f"  std=0 的文件数（velocity 完全一致）: {(velocity_per_file_std == 0).sum()}/{len(velocity_per_file_std)}")
    
    if velocity_per_file_std.mean() < 1:
        log(f"\n  >>> velocity 几乎是常数，对分类无帮助")
    elif velocity_per_file_std.mean() < 5:
        log(f"\n  >>> velocity 变化很小，对分类帮助有限")
    else:
        log(f"\n  >>> velocity 有一定变化，可能有用")
    
    # ========================================================
    # 分析 2：各特征组的方差检查（接近 0 = 没信息）
    # ========================================================
    log("\n" + "=" * 70)
    log("分析 2: 各特征组的整体方差")
    log("=" * 70)
    log("（方差越接近 0，特征越没区分度）\n")
    
    log(f"{'特征组':>12} | {'维度':>5} | {'平均方差':>10} | {'最大方差':>10} | {'最小方差':>10}")
    log("-" * 65)
    for name, arr in all_groups.items():
        variances = arr.var(axis=0)
        log(f"{name:>12} | {arr.shape[1]:>5} | {variances.mean():>10.6f} | "
            f"{variances.max():>10.6f} | {variances.min():>10.6f}")
    
    # ========================================================
    # 分析 3：ANOVA F-score——每个特征对 composer 标签的区分能力
    # ========================================================
    log("\n" + "=" * 70)
    log("分析 3: ANOVA F-score（每个特征组对 composer 的区分能力）")
    log("=" * 70)
    log("（F 越大 = 该特征在不同 composer 间分布差异越大 = 越有判别力）\n")
    
    log(f"{'特征组':>12} | {'平均 F':>10} | {'最大 F':>10} | {'有用特征数 (F>10)':>20}")
    log("-" * 65)
    for name, arr in all_groups.items():
        F, _ = f_classif(arr, y)
        # 替换 nan/inf
        F = np.nan_to_num(F, nan=0, posinf=0, neginf=0)
        useful = (F > 10).sum()
        log(f"{name:>12} | {F.mean():>10.2f} | {F.max():>10.2f} | {useful:>20}")
    
    # ========================================================
    # 分析 4：每个特征组单独训练 LR 的 CV 准确率
    # ========================================================
    log("\n" + "=" * 70)
    log("分析 4: 每个特征组单独的 CV 准确率（LR, 5-fold, 不加 class_weight）")
    log("=" * 70)
    log("（这是最终验证：这个特征组单独有多大用）\n")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    log(f"{'特征组':>12} | {'维度':>5} | {'CV acc':>10}")
    log("-" * 35)
    
    cv_results = {}
    for name, arr in all_groups.items():
        scores = []
        for tr, va in skf.split(arr, y):
            sc = StandardScaler()
            Xtr, Xva = sc.fit_transform(arr[tr]), sc.transform(arr[va])
            m = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
            m.fit(Xtr, y[tr])
            scores.append(accuracy_score(y[va], m.predict(Xva)))
        cv_results[name] = np.mean(scores)
        log(f"{name:>12} | {arr.shape[1]:>5} | {np.mean(scores):>10.4f}")
    
    # ========================================================
    # 分析 5：组合实验
    # ========================================================
    log("\n" + "=" * 70)
    log("分析 5: 组合实验（看哪些组合最强）")
    log("=" * 70)
    
    combos = [
        ("ALL (基线)",                       list(all_groups.keys())),
        ("chroma + interval (教授极简版)",     ['chroma', 'interval']),
        ("chroma + interval + transition",  ['chroma', 'interval', 'transition']),
        ("chroma + interval + duration",    ['chroma', 'interval', 'duration']),
        ("chroma + interval + ioi",         ['chroma', 'interval', 'ioi']),
        ("c + i + dur + ioi (无 transition)", ['chroma', 'interval', 'duration', 'ioi']),
        ("c + i + dur + ioi + transition",   ['chroma', 'interval', 'duration', 'ioi', 'transition']),
        ("ALL - pitch (去掉绝对 pitch)",       ['chroma', 'interval', 'transition', 'duration', 'velocity', 'ioi', 'scalars']),
        ("ALL - pitch - velocity",          ['chroma', 'interval', 'transition', 'duration', 'ioi', 'scalars']),
        ("ALL - pitch - velocity - trans",  ['chroma', 'interval', 'duration', 'ioi', 'scalars']),
    ]
    
    log(f"\n{'组合':>50} | {'维度':>5} | {'CV acc':>10}")
    log("-" * 75)
    
    for combo_name, group_names in combos:
        X = np.concatenate([all_groups[n] for n in group_names], axis=1)
        scores = []
        for tr, va in skf.split(X, y):
            sc = StandardScaler()
            Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
            m = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
            m.fit(Xtr, y[tr])
            scores.append(accuracy_score(y[va], m.predict(Xva)))
        log(f"{combo_name:>50} | {X.shape[1]:>5} | {np.mean(scores):>10.4f}")
    
    # 保存日志
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(_log) + '\n')
    log(f"\n[✓] 日志已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()