"""扫描仓库中「只有定义、无其它引用」的函数与类（只报告，不删除）。

判定：AST 收集定义；全仓库 .py 文本用标识符 \\bname\\b 计数引用。
方法额外检查 self.name / Cls.name；若存在则不算死代码。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = ("src", "tests", "scripts")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
DUNDER_KEEP = {
    "__init__",
    "__new__",
    "__call__",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__iter__",
    "__next__",
    "__len__",
    "__getitem__",
    "__setitem__",
    "__contains__",
    "__repr__",
    "__str__",
    "__eq__",
    "__hash__",
}
# 框架/约定入口：按名字调用，源码中往往看不到显式引用
FRAMEWORK_METHOD_NAMES = {
    "forward",  # torch.nn.Module
}
SB3_ON_PREFIXES = ("_on_",)  # Stable-Baselines3 BaseCallback
AST_VISIT_PREFIX = "visit_"


@dataclass(frozen=True)
class DefInfo:
    kind: str  # function | method | class
    name: str
    file: str
    line: int
    class_name: str | None
    private: bool


def _iter_py_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            files.append(root.resolve())
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            files.append(path.resolve())
    return sorted(set(files))


def _collect_defs(path: Path, repo: Path) -> list[DefInfo]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    rel = str(path.relative_to(repo)).replace("\\", "/")
    out: list[DefInfo] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            out.append(
                DefInfo(
                    kind="class",
                    name=node.name,
                    file=rel,
                    line=node.lineno,
                    class_name=None,
                    private=node.name.startswith("_") and not node.name.startswith("__"),
                )
            )
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._add_fn(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._add_fn(node)
            self.generic_visit(node)

        def _add_fn(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            in_class = bool(self.class_stack)
            out.append(
                DefInfo(
                    kind="method" if in_class else "function",
                    name=node.name,
                    file=rel,
                    line=node.lineno,
                    class_name=self.class_stack[-1] if in_class else None,
                    private=node.name.startswith("_") and not node.name.startswith("__"),
                )
            )

    Visitor().visit(tree)
    return out


def _has_main_guard(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return bool(re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', text))


def _count_ident(text: str, name: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\b", text))


def _method_self_refs(text: str, name: str, class_name: str | None) -> int:
    n = len(re.findall(rf"\bself\.{re.escape(name)}\b", text))
    n += len(re.findall(rf"\bcls\.{re.escape(name)}\b", text))
    if class_name:
        n += len(re.findall(rf"\b{re.escape(class_name)}\.{re.escape(name)}\b", text))
    return n


def _should_skip(defn: DefInfo, file_texts: dict[str, str], repo: Path) -> str | None:
    if defn.name in DUNDER_KEEP or (
        defn.name.startswith("__") and defn.name.endswith("__")
    ):
        return "dunder"
    if defn.name.startswith("test_"):
        return "pytest_test"
    if defn.kind == "function" and defn.name == "main":
        abs_path = (repo / defn.file).resolve()
        if _has_main_guard(abs_path):
            return "script_main"
    # conftest fixtures: top-level functions in conftest.py (heuristic)
    if defn.file.endswith("conftest.py") and defn.kind == "function":
        return "conftest"
    if defn.kind == "method":
        if defn.name in FRAMEWORK_METHOD_NAMES:
            return "framework_method"
        if defn.name.startswith(SB3_ON_PREFIXES):
            return "sb3_callback"
        if defn.name.startswith(AST_VISIT_PREFIX):
            return "ast_visitor"
    return None


def find_unused(
    roots: list[Path],
    *,
    private_only: bool = False,
    repo: Path = REPO_ROOT,
) -> list[dict]:
    py_files = _iter_py_files(roots)
    file_texts: dict[str, str] = {}
    for path in py_files:
        rel = str(path.relative_to(repo)).replace("\\", "/")
        try:
            file_texts[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

    all_text = "\n".join(file_texts.values())
    defs: list[DefInfo] = []
    for path in py_files:
        defs.extend(_collect_defs(path, repo))

    # How many times each name is defined (same name can appear in many files)
    def_counts: dict[str, int] = {}
    for d in defs:
        def_counts[d.name] = def_counts.get(d.name, 0) + 1

    candidates: list[dict] = []
    for d in defs:
        if private_only and not d.private:
            continue
        skip = _should_skip(d, file_texts, repo)
        if skip:
            continue

        total = _count_ident(all_text, d.name)
        # Each definition contributes at least one occurrence of the name.
        # If total <= number of same-named definitions, there is no external use.
        if total > def_counts[d.name]:
            continue

        if d.kind == "method":
            own = file_texts.get(d.file, "")
            if _method_self_refs(own, d.name, d.class_name) > 0:
                continue
            # Also check other files for Class.name / imported usage of method name alone
            # already covered by total > def_counts; methods often only appear as def name.

        item = asdict(d)
        item["occurrences"] = total
        item["same_name_defs"] = def_counts[d.name]
        item["qualname"] = (
            f"{d.class_name}.{d.name}" if d.class_name else d.name
        )
        candidates.append(item)

    candidates.sort(key=lambda x: (x["file"], x["line"]))
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find functions/classes defined but never referenced elsewhere."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Directories (relative to repo root) to scan (default: src tests scripts)",
    )
    parser.add_argument(
        "--private-only",
        action="store_true",
        help="Only report names starting with a single underscore",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table",
    )
    args = parser.parse_args(argv)

    roots = [(REPO_ROOT / r).resolve() for r in args.roots]
    candidates = find_unused(roots, private_only=args.private_only, repo=REPO_ROOT)

    if args.json:
        print(json.dumps(candidates, indent=2, ensure_ascii=False))
    else:
        print(
            "Unused definition candidates "
            f"(AST defs + \\bname\\b refs; private_only={args.private_only})"
        )
        print(
            "Note: string/dynamic dispatch and cross-language refs are not analyzed.\n"
        )
        if not candidates:
            print("No candidates found.")
        else:
            print(f"{'kind':<8} {'line':>5}  {'occ':>3}  qualname  @ file")
            print("-" * 72)
            for c in candidates:
                print(
                    f"{c['kind']:<8} {c['line']:>5}  {c['occurrences']:>3}  "
                    f"{c['qualname']}  @ {c['file']}"
                )
            print(f"\nTotal: {len(candidates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
