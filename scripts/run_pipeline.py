#!/usr/bin/env python3
"""Run the end-to-end history-conditioned imitation training pipeline."""

from __future__ import annotations

import traceback
from typing import Callable, List, Tuple

from pipeline_config import DATASET_TAG, ENABLE_STAGE0_MERGE, TARGET_USERNAME, player_pgn_dir, raw_pgn

import script0_merge_player_pgns as stage0
import script1_parse_pgn_to_positions as stage1
import script2_encode_policy_samples as stage2
import script4_finetune_history_policy as stage3


Stage = Tuple[str, Callable[[], None]]


def run_stage(name: str, fn: Callable[[], None]) -> None:
    """Run one pipeline stage and print start/end markers."""
    print(f"\n[START] {name}")
    fn()
    print(f"[DONE]  {name}")


def main() -> None:
    """Execute configured stages in order."""
    print("=== History-Conditioned Chess Imitation Pipeline ===")
    print(f"Dataset tag: {DATASET_TAG}")
    print(f"Target user: {TARGET_USERNAME}")
    print(f"Merge stage: {'ON' if ENABLE_STAGE0_MERGE else 'OFF'}")
    if ENABLE_STAGE0_MERGE:
        print(f"Player PGN dir: {player_pgn_dir(DATASET_TAG)}")
    print(f"PGN path: {raw_pgn(DATASET_TAG)}")

    stages: List[Stage] = []
    if ENABLE_STAGE0_MERGE:
        stages.append(("Stage 0/3: Merge player PGNs", stage0.main))
    stages.extend(
        [
            ("Stage 1/3: Parse target moves with history", stage1.main),
            ("Stage 2/3: Encode policy samples", stage2.main),
            ("Stage 3/3: Fine-tune history policy", stage3.main),
        ]
    )

    try:
        for name, fn in stages:
            run_stage(name, fn)
    except Exception as exc:
        print("\n[FAILED] Pipeline stopped due to error")
        print(f"Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise

    print(f"\n[SUCCESS] Pipeline completed for dataset: {DATASET_TAG}")


if __name__ == "__main__":
    main()
