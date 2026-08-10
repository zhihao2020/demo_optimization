"""Generate HMSD algorithm structure figure via academic-figure-generator ImageService.

Requires NANOBANANA_API_KEY (and optional NANOBANANA_API_BASE) in the environment
or in external_tool/academic-figure-generator-main/.env

Usage:
  python scripts/gen_hmsd_algorithm_figure.py
  python scripts/gen_hmsd_algorithm_figure.py --resolution 2K --aspect 16:9
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AFG = ROOT / "external_tool" / "academic-figure-generator-main"
PROMPT_MD = ROOT / "Paper" / "figures" / "prompts" / "hmsd_algorithm_structure_prompt.md"
OUT_DIR = ROOT / "Paper" / "figures"


def _load_dotenv() -> None:
    for p in (AFG / ".env", ROOT / ".env"):
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _extract_prompt(md: str) -> str:
    """Extract first fenced code block after English prompt heading, else first ``` block."""
    fences: list[str] = []
    i = 0
    lines = md.splitlines()
    while i < len(lines):
        if lines[i].strip().startswith("```"):
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            fences.append("\n".join(buf).strip())
        i += 1
    if not fences:
        raise RuntimeError(f"No fenced prompt found in {PROMPT_MD}")
    # Prefer the long English diagram prompt
    fences.sort(key=len, reverse=True)
    return fences[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", default="2K", choices=["1K", "2K", "4K"])
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--tag", default="hmsd_afg", help="output filename stem suffix")
    args = parser.parse_args()

    _load_dotenv()
    if not os.environ.get("NANOBANANA_API_KEY"):
        print(
            "ERROR: NANOBANANA_API_KEY not set.\n"
            "  1) Copy external_tool/academic-figure-generator-main/.env.example to .env\n"
            "  2) Set NANOBANANA_API_KEY=...\n"
            "  3) Re-run this script.\n"
            "Fallback: the agent can generate via Imagine using the same prompt file.",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(AFG / "backend"))
    from app.services.image_service import ImageService  # type: ignore

    prompt = _extract_prompt(PROMPT_MD.read_text(encoding="utf-8"))
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Calling ImageService resolution={args.resolution} aspect={args.aspect} ...")

    svc = ImageService()
    result = svc.generate_image(
        prompt=prompt,
        resolution=args.resolution,
        aspect_ratio=args.aspect,
    )
    raw = base64.b64decode(result["image_base64"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"fig_algorithm_{args.tag}.png"
    out_png.write_bytes(raw)
    print(f"Saved: {out_png} ({len(raw):,} bytes, {result['width']}x{result['height']}, {result['duration_ms']} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
