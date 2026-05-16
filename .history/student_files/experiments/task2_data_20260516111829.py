# explore_task2.py
"""先看一下 Task 2 数据长啥样"""
import os

DATAROOT = "student_files/task2_next_sequence_prediction"
OUTPUT_FILE = "task2_exploration.txt"

_log = []
def log(m=""):
    print(m); _log.append(str(m))

with open(os.path.join(DATAROOT, "train.json")) as f:
    train = eval(f.read())
with open(os.path.join(DATAROOT, "test.json")) as f:
    test = eval(f.read())

log("=" * 70)
log("基本规模")
log("=" * 70)
log(f"训练: {len(train)} (type={type(train).__name__})")
log(f"测试: {len(test)} (type={type(test).__name__})")

log(f"\n训练 key 示例（前 3 个）:")
for k in list(train.keys())[:3]:
    log(f"  key = {k!r}")
    log(f"  value = {train[k]!r}")

log(f"\n测试 key 示例（前 3 个）:")
if isinstance(test, dict):
    for k in list(test.keys())[:3]:
        log(f"  key = {k!r}")
else:
    for k in test[:3]:
        log(f"  item = {k!r}")

# 标签分布
if isinstance(train, dict):
    from collections import Counter
    label_counts = Counter(train.values())
    log(f"\n标签分布:")
    for lbl, cnt in label_counts.items():
        log(f"  {lbl!r}: {cnt} ({cnt/len(train)*100:.2f}%)")

# 看看实际 MIDI 文件
log(f"\n{'='*70}\n采样一个文件看格式\n{'='*70}")
import miditoolkit
first_key = list(train.keys())[0]
log(f"key = {first_key!r}")
log(f"key 类型: {type(first_key).__name__}")
log(f"value: {train[first_key]}")

# 如果 key 是 tuple，访问两个 path
if isinstance(first_key, tuple):
    p1, p2 = first_key
    log(f"\nPath 1: {p1}")
    log(f"Path 2: {p2}")
    full_p1 = os.path.join(DATAROOT, p1)
    full_p2 = os.path.join(DATAROOT, p2)
    log(f"Path 1 exists: {os.path.exists(full_p1)}")
    log(f"Path 2 exists: {os.path.exists(full_p2)}")
    
    midi1 = miditoolkit.midi.parser.MidiFile(full_p1)
    midi2 = miditoolkit.midi.parser.MidiFile(full_p2)
    log(f"\nMIDI 1: instruments={len(midi1.instruments)}, n_notes(inst0)={len(midi1.instruments[0].notes)}")
    log(f"MIDI 2: instruments={len(midi2.instruments)}, n_notes(inst0)={len(midi2.instruments[0].notes)}")
    log(f"MIDI 1 时长: start={midi1.instruments[0].notes[0].start}, end={midi1.instruments[0].notes[-1].end}")
    log(f"MIDI 2 时长: start={midi2.instruments[0].notes[0].start}, end={midi2.instruments[0].notes[-1].end}")

# 文件总数（统计不同 path 的数量）
all_paths = set()
if isinstance(train, dict):
    for k in train.keys():
        if isinstance(k, tuple):
            all_paths.add(k[0])
            all_paths.add(k[1])
        else:
            all_paths.add(k)
log(f"\n训练中涉及的不同 MIDI 文件数: {len(all_paths)}")

with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(_log) + '\n')
print(f"\n[✓] {OUTPUT_FILE}")