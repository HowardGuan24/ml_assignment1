# statistics/task1_baseline_format_check.py
"""
用基线方法（average pitch + average duration）生成 predictions1.json
目的：得到一个 autograder 能正确读取的"标准格式样例"，和我们的对比
"""
import os
import numpy as np
import miditoolkit
from sklearn.linear_model import LogisticRegression

DATAROOT = "student_files/task1_composer_classification"
OUT_PATH = "statistics/predictions1_baseline.json"

def features(path):
    midi_obj = miditoolkit.midi.parser.MidiFile(DATAROOT + '/' + path)
    notes = midi_obj.instruments[0].notes
    num_notes = len(notes)
    avg_pitch = sum(n.pitch for n in notes) / num_notes
    avg_dur = sum(n.end - n.start for n in notes) / num_notes
    return [avg_pitch, avg_dur]

with open(DATAROOT + "/train.json") as f:
    train = eval(f.read())
with open(DATAROOT + "/test.json") as f:
    test = eval(f.read())

print("提取训练特征...")
X_tr = [features(k) for k in train]
y_tr = [int(train[k]) for k in train]

print("训练模型...")
model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)

print("预测测试集...")
predictions = {}
for k in test:
    x = features(k)
    pred = model.predict([x])
    predictions[k] = int(pred[0])

# 完全按 baseline 的写法
with open(OUT_PATH, "w") as f:
    f.write(repr(predictions) + '\n')

print(f"[✓] 写入 {OUT_PATH}")
print(f"预测数: {len(predictions)}")

# 打印前几个 entry 和最后几个 entry，方便对比
items = list(predictions.items())
print("\n前 3 条 entry:")
for k, v in items[:3]:
    print(f"  key={k!r}, value={v!r} (type={type(v).__name__})")
print("\n最后 3 条 entry:")
for k, v in items[-3:]:
    print(f"  key={k!r}, value={v!r}")

# 打印文件开头和结尾几个字符（看格式）
with open(OUT_PATH) as f:
    content = f.read()
print(f"\n文件长度: {len(content)} chars")
print(f"开头 200 字符: {content[:200]!r}")
print(f"结尾 100 字符: {content[-100:]!r}")