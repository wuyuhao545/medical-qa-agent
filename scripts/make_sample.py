# scripts/make_sample.py
"""从完整数据集中抽取前 N 条作为 demo 样本"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "medical_knowledge" / "articles.jsonl"
DST = ROOT / "data" / "medical_knowledge" / "sample.jsonl"
N = 100

if not SRC.exists():
    print(f"❌ 源文件不存在: {SRC}")
    raise SystemExit(1)

count = 0
with SRC.open("r", encoding="utf-8") as f_in, DST.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        if count >= N:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if obj.get("categories"):          # 优先挑带分类的
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                count += 1
        except json.JSONDecodeError:
            continue

print(f"✅ 已生成 {count} 条样本 → {DST}")
print(f"   文件大小: {DST.stat().st_size / 1024:.1f} KB")