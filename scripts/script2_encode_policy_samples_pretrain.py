#!/usr/bin/env python3
"""Encode multi-player pretraining rows into policy samples."""

from __future__ import annotations

from pipeline_config import HISTORY_PLIES, PRETRAIN_DATASET_TAG, processed_dir
import script2_encode_policy_samples as encoder


encoder.CONFIG.update(
    {
        "dataset_tag": PRETRAIN_DATASET_TAG,
        "history_plies": HISTORY_PLIES,
        "input_jsonl": processed_dir(PRETRAIN_DATASET_TAG) / "positions_history.jsonl",
        "output_jsonl": processed_dir(PRETRAIN_DATASET_TAG) / "policy_samples.jsonl",
        "progress_every": 10000,
    }
)


def main() -> None:
    """Run stage 2 encoding for the multi-player pretraining dataset."""
    encoder.main()


if __name__ == "__main__":
    main()
