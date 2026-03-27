#!/usr/bin/env python3
"""Merge per-month PGN files into one dataset PGN."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pipeline_config import (
    DATASET_TAG,
    PLAYER_PGN_GLOB,
    PLAYER_PGN_RECURSIVE,
    player_pgn_dir,
    raw_pgn,
)


def collect_pgn_files(folder: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(p for p in folder.rglob(pattern) if p.is_file())
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def merge_pgn_files(files: List[Path], output_path: Path) -> None:
    chunks: List[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            chunks.append(text)

    if not chunks:
        raise ValueError("No non-empty PGN files found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(chunks).strip() + "\n", encoding="utf-8")

    print(f"Merged {len(chunks)} files -> {output_path}")


def main() -> None:
    in_dir = player_pgn_dir(DATASET_TAG)
    if not in_dir.exists():
        raise FileNotFoundError(f"Player PGN folder not found: {in_dir}")

    files = collect_pgn_files(in_dir, PLAYER_PGN_GLOB, PLAYER_PGN_RECURSIVE)
    if not files:
        raise FileNotFoundError(f"No PGN files found in {in_dir}")

    merge_pgn_files(files, raw_pgn(DATASET_TAG))


if __name__ == "__main__":
    main()
