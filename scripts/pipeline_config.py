#!/usr/bin/env python3
"""Central configuration and path helpers for the imitation pipeline."""

from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Active target-player dataset.
DATASET_TAG = "Bobby_Fischer"
TARGET_USERNAME = "Fischer Robert J (USA)"
# Optional aliases for robust name matching in PGN headers.
TARGET_NAME_ALIASES: list[str] = ['Fischer Robert J (USA)', 'Fischer', "Robert James Fischer", "Fischer, Robert J", "Robert J FISCHER"]

# Shared pool for representation pretraining.
PRETRAIN_DATASET_TAG = "pretrain_multi"
PRETRAIN_MERGED_FILENAME = "pretrain_multi_merged.pgn"

# Dedicated root for target-player fine-tuning datasets.
FINETUNE_ROOT_NAME = "finetune_players"

# Optional stage 0 merge settings.
ENABLE_STAGE0_MERGE = True
PLAYER_PGN_GLOB = "*.pgn"
PLAYER_PGN_RECURSIVE = True

# History window for sequence features.
HISTORY_PLIES = 8

# Stage 3 pretraining mode in run_pretrain_pipeline.py.
# False: in-memory pretrain script (faster, higher RAM)
# True: streamed pretrain script (lower RAM, full-data friendly)
USE_STREAM_PRETRAIN_STAGE3 = True


def raw_dir() -> Path:
    return PROJECT_ROOT / "data" / "raw"


def pretrain_raw_dir() -> Path:
    return raw_dir() / PRETRAIN_DATASET_TAG


def pretrain_player_dir(player_slug: str) -> Path:
    return pretrain_raw_dir() / player_slug


def pretrain_player_pgn(player_slug: str) -> Path:
    return pretrain_raw_dir() / f"{player_slug}.pgn"


def pretrain_merged_pgn() -> Path:
    return pretrain_raw_dir() / PRETRAIN_MERGED_FILENAME


def finetune_raw_root() -> Path:
    return raw_dir() / FINETUNE_ROOT_NAME


def raw_pgn(tag: str = DATASET_TAG) -> Path:
    return finetune_raw_root() / f"{tag}.pgn"


def player_pgn_dir(tag: str = DATASET_TAG) -> Path:
    return finetune_raw_root() / tag


def processed_dir(tag: str = DATASET_TAG) -> Path:
    return PROJECT_ROOT / "data" / "processed" / tag


def models_dir(tag: str = DATASET_TAG) -> Path:
    return PROJECT_ROOT / "models" / tag


def outputs_dir(tag: str = DATASET_TAG) -> Path:
    return PROJECT_ROOT / "outputs" / tag
