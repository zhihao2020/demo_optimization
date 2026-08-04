import json
from pathlib import Path
root = Path(r"D:\Code\0622\optimal_demo\reference_papers\md")
for d in sorted(root.iterdir()):
    if not d.is_dir() or not d.name.endswith("_md"):
        continue
    man = d / "manifest.json"
    doc = d / "document.md"
    title = d.name
    abs_prev = ""
    if man.is_file():
        m = json.loads(man.read_text(encoding="utf-8"))
        title = m.get("title") or title
        abs_prev = (m.get("abstract_preview") or "")[:280]
    size = doc.stat().st_size if doc.is_file() else 0
    print("="*80)
    print("DIR:", d.name)
    print("TITLE:", title)
    print("SIZE:", size)
    print("ABS:", abs_prev)
