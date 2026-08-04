import re, subprocess, sys, time
from pathlib import Path

ROOT = Path(r"D:\Code\0622\optimal_demo\reference_papers")
CONVERT = Path(r"C:\Users\xuzhihao\.claude\skills\mineru-pdf-reader\scripts\convert_pdf.py")
OUT_ROOT = ROOT / "md"
OUT_ROOT.mkdir(exist_ok=True)

def safe_stem(name: str, max_len: int = 80) -> str:
    s = Path(name).stem
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "paper"

pdfs = sorted(ROOT.glob("*.pdf"))
print(f"found {len(pdfs)} pdfs", flush=True)
results = []
for i, pdf in enumerate(pdfs, 1):
    stem = safe_stem(pdf.name)
    out_dir = OUT_ROOT / f"{stem}_md"
    doc = out_dir / "document.md"
    man = out_dir / "manifest.json"
    if doc.is_file() and doc.stat().st_size > 500:
        print(f"[{i}/{len(pdfs)}] SKIP exists {stem}", flush=True)
        results.append((pdf.name, "skip", str(out_dir)))
        continue
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{i}/{len(pdfs)}] CONVERT {pdf.name}", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(CONVERT), str(pdf), str(out_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        ok = proc.returncode == 0 and (doc.is_file() or man.is_file())
        status = "ok" if ok else f"fail_rc={proc.returncode}"
        if not ok:
            print(proc.stdout[-1500:] if proc.stdout else "", flush=True)
            print(proc.stderr[-1500:] if proc.stderr else "", flush=True)
        else:
            size = doc.stat().st_size if doc.is_file() else 0
            print(f"  -> {status} {time.time()-t0:.1f}s document.md={size}B", flush=True)
        results.append((pdf.name, status, str(out_dir)))
    except subprocess.TimeoutExpired:
        print("  -> timeout", flush=True)
        results.append((pdf.name, "timeout", str(out_dir)))
    except Exception as e:
        print(f"  -> error {e}", flush=True)
        results.append((pdf.name, f"error:{e}", str(out_dir)))

print("\n=== SUMMARY ===", flush=True)
for name, st, od in results:
    print(f"{st:12} {name}", flush=True)
fail = [r for r in results if r[1] not in ("ok", "skip")]
print(f"ok/skip={len(results)-len(fail)} fail={len(fail)}", flush=True)
sys.exit(1 if fail else 0)
