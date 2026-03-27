#!/usr/bin/env python3
"""Run multi-player pretraining pipeline: merge, parse, encode, pretrain."""

from __future__ import annotations

import traceback
from typing import Callable, List, Tuple

import script0_merge_pretrain_pgns as stage0
import script1_parse_multi_player_positions as stage1
import script2_encode_policy_samples_pretrain as stage2
import script3_pretrain_history_policy as stage3
import script3_pretrain_history_policy_stream as stage3_stream
from pipeline_config import USE_STREAM_PRETRAIN_STAGE3


Stage = Tuple[str, Callable[[], None]]


def run_stage(name: str, fn: Callable[[], None]) -> None:
    """Run one stage and print start/end markers."""
    print(f"\n[START] {name}")
    fn()
    print(f"[DONE]  {name}")


def main() -> None:
    """Execute full multi-player pretraining pipeline."""
    stage3_name = "Stage 3/4: Pretrain representation model"
    stage3_fn = stage3_stream.main if USE_STREAM_PRETRAIN_STAGE3 else stage3.main
    stage3_mode = "stream" if USE_STREAM_PRETRAIN_STAGE3 else "standard"
    print(f"[INFO] Pretrain Stage 3 mode: {stage3_mode}")

    stages: List[Stage] = [
        ("Stage 0/4: Merge pretrain PGNs", stage0.main),
        ("Stage 1/4: Parse multi-player positions", stage1.main),
        ("Stage 2/4: Encode policy samples", stage2.main),
        (stage3_name, stage3_fn),
    ]

    try:
        for name, fn in stages:
            run_stage(name, fn)
    except Exception as exc:
        print("\n[FAILED] Pretraining pipeline stopped")
        print(f"Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise

    print("\n[SUCCESS] Pretraining pipeline completed")


if __name__ == "__main__":
    main()
