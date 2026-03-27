#!/usr/bin/env python3
"""Merge many pretraining PGNs into one merged pretraining PGN file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

from pipeline_config import PRETRAIN_MERGED_FILENAME, pretrain_merged_pgn, pretrain_raw_dir


CONFIG = {
    "input_root": pretrain_raw_dir(),
    "glob": "*.pgn",
    "recursive": True,
    "exclude_subdirs": ["db_pages"],
    "output_path": pretrain_merged_pgn(),
    "verbose": True,
}


def should_skip(path: Path, output_path: Path, exclude_subdirs: List[str]) -> bool:
    """Return true if input file should be skipped for merge."""
    if path.resolve() == output_path.resolve():
        return True

    parts = {p.lower() for p in path.parts}
    for sub in exclude_subdirs:
        if sub.lower() in parts:
            return True
    return False


def collect_inputs(root: Path, pattern: str, recursive: bool, output_path: Path, exclude_subdirs: List[str]) -> List[Path]:
    """Collect candidate pretraining PGN input files."""
    if recursive:
        files = sorted(p for p in root.rglob(pattern) if p.is_file())
    else:
        files = sorted(p for p in root.glob(pattern) if p.is_file())

    return [p for p in files if not should_skip(p, output_path, exclude_subdirs)]


def merge_files(inputs: List[Path], output_path: Path) -> int:
    """Merge files with chunk-level deduplication and return kept chunk count."""
    seen = set()
    chunks: List[str] = []

    for path in inputs:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        chunks.append(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = "\n\n".join(chunks).strip()
    output_path.write_text((merged + "\n") if merged else "", encoding="utf-8")
    return len(chunks)


def main() -> None:
    """Merge pretraining PGNs into one canonical merged file."""
    root = Path(CONFIG["input_root"])
    output_path = Path(CONFIG["output_path"])

    if not root.exists():
        raise FileNotFoundError(f"Pretrain root not found: {root}")

    inputs = collect_inputs(
        root=root,
        pattern=str(CONFIG["glob"]),
        recursive=bool(CONFIG["recursive"]),
        output_path=output_path,
        exclude_subdirs=list(CONFIG.get("exclude_subdirs", [])),
    )
    if not inputs:
        raise FileNotFoundError(f"No PGN files found under {root} to merge")

    kept = merge_files(inputs, output_path)

    if bool(CONFIG.get("verbose", True)):
        print(f"Pretrain input files scanned: {len(inputs)}")
        print(f"Pretrain chunks merged:      {kept}")
        print(f"Merged pretrain PGN:        {output_path}")
        print(f"Merged filename tag:        {PRETRAIN_MERGED_FILENAME}")


if __name__ == "__main__":
    main()
