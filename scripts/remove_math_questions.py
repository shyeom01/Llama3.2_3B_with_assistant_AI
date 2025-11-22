"""remove_math_questions.py

Utility script to remove math-related samples from JSON dataset files.

- Each input JSON file must contain a list of objects (dicts).
- The script removes any example whose *question* field appears to be mathematical and writes the filtered list to a new file with the suffix '_nomath.json' (placed in the same directory as the source file).
- If no non-math examples remain, no file is written for that input.

Usage (from project root):

    python scripts/remove_math_questions.py extracted_dataset/train/

or for multiple folders:

    python scripts/remove_math_questions.py extracted_dataset/train/ extracted_dataset/test/

After running, look for new files like 'reduced_openmath_train_SFT_pairs_nomath.json' in the same folders. These contain the data with all math questions removed.

At the end, a summary of all files processed is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Any

# --------------------------- math‑detection logic --------------------------- #

# Pre‑compile regex patterns for speed.
# Only include keywords that are highly specific to strict math problems.
# Removed broad terms like 'calculate', 'simplify', 'compute', 'evaluate', 'number', 'expression', 'function', 'variable'.
_MATH_KEYWORDS = [
    r"solve", r"integral", r"derivative", r"equation", r"probability", r"geometry", r"algebra",
    r"sum", r"product", r"difference", r"quotient", r"matrix", r"vector", r"limit", r"series",
    r"angle", r"triangle", r"rectangle", r"circle", r"radius", r"area", r"volume", r"perimeter",
    r"graph", r"polynomial", r"math", r"mathematics"
]

# Typical LaTeX/math symbols or delimiters.
_MATH_SYMBOLS = [
    r"\\frac", r"\\sum", r"\\int", r"\\sqrt", r"\\lim", r"\\log", r"\\sin", r"\\cos",
    r"\\tan", r"\\theta", r"\\pi", r"\\infty", r"\\partial", r"\\nabla",
    r"[=<>±×÷∑∫√∞πθ]",  # basic unicode symbols inside [] so each is matched separately
]

# Combine into a single regex pattern (ignore case).
_MATH_REGEX = re.compile(
    r"|".join(_MATH_KEYWORDS + _MATH_SYMBOLS), re.IGNORECASE
)


def is_math_question(text: str) -> bool:
    """Return True if *text* looks like a math problem/question or bare equation."""
    if not text:
        return False
    # Math keywords or symbols
    if _MATH_REGEX.search(text):
        return True
    # If text is just a number or equation (e.g., '2x+3=5', '42', etc.)
    if re.fullmatch(r"[\d\s\+\-\*/=\^\(\)\[\]\.\,xXyYzZaAbBcC]+", text.strip()):
        return True
    return False


# ------------------------------ I/O helpers -------------------------------- #

QuestionKeys = [
    "sft_question",  # openmath/shp pairs
    "instruction",   # generic Alpaca‑style format
    "prompt",        # fallback
]


def _extract_question(sample: Dict[str, Any]) -> str:
    """Return the *question* string from a dataset sample.

    Different datasets use different keys.  This helper tries a few common
    ones and falls back to converting the entire sample to a string.
    """
    for key in QuestionKeys:
        if key in sample and isinstance(sample[key], str):
            return sample[key]
    # Fallback: join string values
    return " ".join(str(v) for v in sample.values() if isinstance(v, str))


# --------------------------- core filtering code --------------------------- #

def filter_file(path: Path, keep_nomath: bool = True) -> Path | None:
    """Filter *path* in‑place (writing to *_nomath.json). Returns output path or None."""
    with path.open("r", encoding="utf-8") as f:
        try:
            data: List[Dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping {path} – invalid JSON: {e}")
            return None

    if not isinstance(data, list):
        print(f"[WARN] {path} does not contain a JSON list – skipping.")
        return None

    filtered: List[Dict[str, Any]] = []
    math_cnt = 0
    for sample in data:
        q = _extract_question(sample)
        if is_math_question(q):
            math_cnt += 1
            continue
        filtered.append(sample)

    total = len(data)
    kept = len(filtered)
    print(f"Processed {path}: total={total}, math={math_cnt}, kept={kept}")

    if kept == 0:
        print(f"[INFO] No non‑math samples left in {path}. File not written.")
        return None

    out_path = path.with_name(path.stem + "_nomath" + path.suffix)
    if keep_nomath:
        if out_path.exists():
            print(f"[WARN] Output file {out_path} already exists and will be overwritten.")
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
            print(f"→ Wrote filtered file: {out_path}")
        except PermissionError:
            print(f"[ERROR] Permission denied when writing {out_path}")
            return None
    return out_path


def gather_json_files(inputs: List[str]) -> List[Path]:
    """Expand *inputs* (files/directories/globs) into a list of JSON file paths."""
    files: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(p.rglob("*.json"))
        else:
            # Support wildcards (globs)
            files.extend(Path().glob(item)) if any(c in item for c in "*?[]") else files.append(p)
    # Keep unique / existing JSON files only
    unique_existing = [f for f in sorted(set(files)) if f.is_file() and f.suffix == ".json"]
    return unique_existing


# --------------------------------- CLI ------------------------------------ #

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Remove math‑related questions from JSON dataset files.")
    parser.add_argument("inputs", nargs="+", help="JSON files or directories/globs to process")
    parser.add_argument("--no-write", action="store_true", help="Do not write filtered files, just report counts")
    args = parser.parse_args(argv)

    json_files = gather_json_files(args.inputs)
    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    total_files = 0
    files_written = 0
    total_removed = 0
    total_kept = 0
    for jf in json_files:
        total_files += 1
        before = None
        try:
            with jf.open("r", encoding="utf-8") as f:
                before = len(json.load(f))
        except Exception:
            before = None
        out = filter_file(jf, keep_nomath=not args.no_write)
        if out is not None:
            files_written += 1
            try:
                with out.open("r", encoding="utf-8") as f:
                    after = len(json.load(f))
                if before is not None:
                    total_removed += before - after
                    total_kept += after
            except Exception:
                pass
    print(f"\nSummary: {total_files} files processed. {files_written} filtered files written.")
    print(f"Total non-math samples kept: {total_kept}, math samples removed: {total_removed}")
    if files_written == 0:
        print("[INFO] No filtered files written. All data may have been math-related or empty.")

if __name__ == "__main__":
    main()
