# explore_task1.py
"""
Task 1 数据探查脚本
跑这个之前，请把 dataroot 改成你本地的路径
"""
import os
import json
from collections import Counter, defaultdict
import numpy as np
import miditoolkit

# ====== 修改这里为你本地的路径 ======
DATAROOT = "/home/howard/ml_assignment1/student_files_updated/student_files/task1_composer_classification"
# ====================================

train_path = os.path.join(DATAROOT, "train.json")
test_path = os.path.join(DATAROOT, "test.json")

with open(train_path) as f:
    train = eval(f.read())
with open(test_path) as f:
    test = eval(f.read())

print("=" * 70)
print("基本规模")
print("=" * 70)
print(f"训练集样本数: {len(train)}")
print(f"测试集样本数: {len(test)}")
print(f"训练/测试比例: {len(train)/len(test):.2f}")

# ---------- 标签分布 ----------
labels = list(train.values())
label_counts = Counter(labels)
print(f"\n{'='*70}\n标签分布（作曲家 ID -> 样本数）\n{'='*70}")
print(f"独立作曲家数: {len(label_counts)}")
for cid, cnt in sorted(label_counts.items()):
    pct = cnt / len(train) * 100
    bar = '█' * int(pct / 2)
    print(f"  composer {cid}: {cnt:5d} ({pct:5.2f}%) {bar}")
mx, mn = max(label_counts.values()), min(label_counts.values())
print(f"\n不平衡度: 最多 / 最少 = {mx} / {mn} = {mx/mn:.2f}x")

# ---------- MIDI 文件特征 ----------
print(f"\n{'='*70}\nMIDI 文件特征（采样 100 个查看）\n{'='*70}")
sample_paths = list(train.keys())[:100]

n_notes_list = []
n_instruments_list = []
duration_list = []
pitch_min_list, pitch_max_list = [], []
program_set = set()
multi_instrument_count = 0
errors = []

for path in sample_paths:
    try:
        midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, path))
        n_instruments_list.append(len(midi.instruments))
        if len(midi.instruments) > 1:
            multi_instrument_count += 1
        # 取第一个 instrument 的 notes（和 baseline 保持一致）
        notes = midi.instruments[0].notes
        n_notes_list.append(len(notes))
        if notes:
            pitches = [n.pitch for n in notes]
            pitch_min_list.append(min(pitches))
            pitch_max_list.append(max(pitches))
            duration_list.append(max(n.end for n in notes) - min(n.start for n in notes))
        for inst in midi.instruments:
            program_set.add((inst.program, inst.is_drum))
    except Exception as e:
        errors.append((path, str(e)))

print(f"采样 {len(sample_paths)} 个，读取失败 {len(errors)} 个")
if errors[:3]:
    print(f"前 3 个错误: {errors[:3]}")

print(f"\n每个文件的音符数（基于第一 instrument）:")
print(f"  均值: {np.mean(n_notes_list):.0f}, 中位数: {np.median(n_notes_list):.0f}")
print(f"  最小: {min(n_notes_list)}, 最大: {max(n_notes_list)}")

print(f"\n每个文件的 instrument 数:")
print(f"  均值: {np.mean(n_instruments_list):.2f}, 中位数: {np.median(n_instruments_list):.0f}")
print(f"  最小: {min(n_instruments_list)}, 最大: {max(n_instruments_list)}")
print(f"  含多 instrument 的文件数: {multi_instrument_count}/{len(sample_paths)}")

print(f"\n时长（最后音符结束 - 第一音符开始，单位 tick）:")
print(f"  均值: {np.mean(duration_list):.0f}, 中位数: {np.median(duration_list):.0f}")

print(f"\n音高范围:")
print(f"  全局最低音: {min(pitch_min_list)}, 全局最高音: {max(pitch_max_list)}")
print(f"  平均最低: {np.mean(pitch_min_list):.1f}, 平均最高: {np.mean(pitch_max_list):.1f}")

print(f"\n出现的乐器（program, is_drum） TOP 10:")
prog_counter = Counter()
for path in sample_paths:
    try:
        midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, path))
        for inst in midi.instruments:
            prog_counter[(inst.program, inst.is_drum)] += 1
    except Exception:
        pass
for (prog, drum), cnt in prog_counter.most_common(10):
    print(f"  program={prog}, is_drum={drum}: {cnt}")

# ---------- 简单 sanity check：看看一个文件长啥样 ----------
print(f"\n{'='*70}\n样本文件 #0 详细信息\n{'='*70}")
first_path = list(train.keys())[0]
midi = miditoolkit.midi.parser.MidiFile(os.path.join(DATAROOT, first_path))
print(f"文件: {first_path}")
print(f"标签 (composer id): {train[first_path]}")
print(f"ticks_per_beat: {midi.ticks_per_beat}")
print(f"instruments 数: {len(midi.instruments)}")
for i, inst in enumerate(midi.instruments):
    print(f"  instrument {i}: program={inst.program}, is_drum={inst.is_drum}, "
          f"name={inst.name!r}, n_notes={len(inst.notes)}")
notes = midi.instruments[0].notes
print(f"前 5 个 notes:")
for n in notes[:5]:
    print(f"  pitch={n.pitch}, velocity={n.veloci