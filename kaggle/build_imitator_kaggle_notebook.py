#!/usr/bin/env python3
"""Generate a self-contained Kaggle notebook with parallel preprocessing."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT.parent / "scripts"
NOTEBOOK_PATH = ROOT / "imitator_kaggle.ipynb"

EMBED_FILES = [
    "chess_feature_utils.py",
    "history_policy_lib.py",
    "pipeline_config.py",
    "script1_parse_multi_player_positions.py",
    "script1_parse_pgn_to_positions.py",
    "script2_encode_policy_samples.py",
    "script3_pretrain_history_policy.py",
    "script3_pretrain_history_policy_stream.py",
    "script4_finetune_history_policy.py",
]

embedded_files = {}
for name in EMBED_FILES:
    path = SCRIPTS_DIR / name
    embedded_files[f"scripts/{name}"] = base64.b64encode(path.read_bytes()).decode("ascii")


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md_cell(
        """# Imitator on Kaggle: Self-Contained Pipeline With Parallel Preprocessing

This notebook needs only the raw chess dataset. It writes the required project files into `/kaggle/working/Imitator`, discovers the Kaggle raw PGN layout, preprocesses PGNs in parallel on CPU, and then trains the pretrain and fine-tune models.

Expected dataset layout:
- `/kaggle/input/datasets/yukuanzou/raw-chess-data/data/twic/*.pgn`
- `/kaggle/input/datasets/yukuanzou/raw-chess-data/data/target/bobby_fischer/*.pgn`

Preprocessing is CPU-bound. GPU is used only for model training.
"""
    ),
    code_cell(
        """RUN_PRETRAIN = True
RUN_FINETUNE = True
EXPORT_ARTIFACTS = True
EXPORT_PROCESSED = False

INSTALL_COMPAT_TORCH = True
USE_STREAM_PRETRAIN_STAGE3 = True
STRICT_TARGET_ISOLATION = False
PREPROCESS_WORKERS = 2

DATASET_TAG = "Bobby_Fischer"
TARGET_USERNAME = "Fischer Robert J (USA)"
TARGET_NAME_ALIASES = [
    "Fischer Robert J (USA)",
    "Fischer",
    "Robert James Fischer",
    "Fischer, Robert J",
    "Robert J FISCHER",
]

MODEL_PROFILE = "kaggle_large"

PROFILE_CONFIGS = {
    "baseline": {
        "history_plies": 8,
        "feature_embed_dim": 64,
        "history_hidden_dim": 96,
        "shared_hidden_dim": 128,
        "dropout": 0.10,
        "pretrain_epochs": 8,
        "pretrain_batch_size": 256,
        "pretrain_eval_batch_size": 256,
        "pretrain_grad_accum_steps": 1,
        "pretrain_shuffle_buffer_size": 20000,
        "pretrain_train_eval_max_samples": 50000,
        "pretrain_valid_eval_max_samples": 50000,
        "finetune_epochs": 12,
        "finetune_batch_size": 192,
        "finetune_eval_batch_size": 192,
        "finetune_grad_accum_steps": 1,
        "encoder_learning_rate": 3e-4,
        "head_learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
    "kaggle_large": {
        "history_plies": 12,
        "feature_embed_dim": 96,
        "history_hidden_dim": 160,
        "shared_hidden_dim": 256,
        "dropout": 0.12,
        "pretrain_epochs": 10,
        "pretrain_batch_size": 384,
        "pretrain_eval_batch_size": 384,
        "pretrain_grad_accum_steps": 1,
        "pretrain_shuffle_buffer_size": 60000,
        "pretrain_train_eval_max_samples": 100000,
        "pretrain_valid_eval_max_samples": 100000,
        "finetune_epochs": 16,
        "finetune_batch_size": 256,
        "finetune_eval_batch_size": 256,
        "finetune_grad_accum_steps": 1,
        "encoder_learning_rate": 2.5e-4,
        "head_learning_rate": 8e-4,
        "weight_decay": 1e-5,
    },
}

PRETRAIN_LEARNING_RATE = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-5
PRETRAIN_PRINT_EVERY = 25
PRETRAIN_SCAN_PROGRESS_EVERY = 100000
PRETRAIN_STREAM_PROGRESS_EVERY = 200000
FINETUNE_PRINT_EVERY = 20
FINETUNE_USE_PRETRAINED_ENCODERS = True

EXPORT_NAME = "imitator_kaggle_artifacts"
KAGGLE_INPUT_DATA_DIR = "/kaggle/input/datasets/yukuanzou/raw-chess-data"
"""
    ),
    code_cell(
        """if INSTALL_COMPAT_TORCH:
    %pip uninstall -y -q torch torchvision torchaudio
    %pip install -q --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 python-chess
else:
    %pip install -q -U python-chess

import os
import platform
import shutil
import torch


def bytes_to_gb(n: int) -> float:
    return n / (1024 ** 3)


disk = shutil.disk_usage("/kaggle/working")
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print(f"working disk free: {bytes_to_gb(disk.free):.1f} GB")
print("cwd:", os.getcwd())
"""
    ),
    code_cell(
        """import base64
import json
import re
import shutil
from pathlib import Path

EMBEDDED_FILES = """ + json.dumps(embedded_files, indent=2) + """

WORK_ROOT = Path("/kaggle/working/Imitator")
SCRIPTS_DIR = WORK_ROOT / "scripts"
RAW_ROOT = WORK_ROOT / "data" / "raw"
PRETRAIN_ROOT = RAW_ROOT / "pretrain_multi"
FINETUNE_ROOT = RAW_ROOT / "finetune_players"

if WORK_ROOT.exists():
    shutil.rmtree(WORK_ROOT)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
PRETRAIN_ROOT.mkdir(parents=True, exist_ok=True)
FINETUNE_ROOT.mkdir(parents=True, exist_ok=True)

for rel_path, encoded in EMBEDDED_FILES.items():
    out_path = WORK_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(encoded.encode("ascii")))

SHARDED_PRETRAIN_SCRIPT = r' + "

def normalize_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def copy_tree(src: Path, dst: Path) -> int:
    copied = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
        copied += 1
    return copied


def discover_data_root() -> Path:
    explicit = Path(KAGGLE_INPUT_DATA_DIR)
    if explicit.exists():
        return explicit
    fallback = Path("/kaggle/input")
    for data_dir in fallback.rglob("data"):
        if (data_dir / "twic").exists() and (data_dir / "target").exists():
            return data_dir.parent
    raise FileNotFoundError("Could not find raw chess data root under /kaggle/input")


data_dataset_root = discover_data_root()
data_root = data_dataset_root / "data"
print("data dataset root:", data_dataset_root)
print("data root:", data_root)

stats = {"twic_files": 0, "target_files": 0}
if (data_root / "twic").exists():
    stats["twic_files"] = copy_tree(data_root / "twic", PRETRAIN_ROOT)

target_slug = normalize_slug(DATASET_TAG)
if (data_root / "target").exists():
    for child in (data_root / "target").iterdir():
        if child.is_dir() and normalize_slug(child.name) == target_slug:
            stats["target_files"] = copy_tree(child, FINETUNE_ROOT / DATASET_TAG)
            break

print("materialized files:", len(EMBEDDED_FILES))
print("hydration stats:", json.dumps(stats, indent=2))
print("pretrain pgns:", len(list(PRETRAIN_ROOT.rglob('*.pgn'))))
print("target pgns:", len(list((FINETUNE_ROOT / DATASET_TAG).rglob('*.pgn'))))
"""
    ),
    code_cell(
        """import json
import re
from pathlib import Path

profile = PROFILE_CONFIGS[MODEL_PROFILE]
HISTORY_PLIES = int(profile["history_plies"])


def replace_assignment(text: str, name: str, value_literal: str) -> str:
    pattern = rf"^{re.escape(name)}\\s*=\\s*.*$"
    replacement = f"{name} = {value_literal}"
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not patch assignment for {name}")
    return new_text


def replace_config_value(text: str, key: str, value_literal: str) -> str:
    pattern = rf'^(\\s*"{re.escape(key)}":\\s*).*$'
    replacement = rf'\\g<1>{value_literal},'
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not patch CONFIG key {key}")
    return new_text


def patch_pipeline_config() -> None:
    path = WORK_ROOT / "scripts" / "pipeline_config.py"
    text = path.read_text(encoding="utf-8")
    text = replace_assignment(text, "DATASET_TAG", json.dumps(DATASET_TAG))
    text = replace_assignment(text, "TARGET_USERNAME", json.dumps(TARGET_USERNAME))
    text = replace_assignment(text, "TARGET_NAME_ALIASES: list[str]", repr(TARGET_NAME_ALIASES))
    text = replace_assignment(text, "ENABLE_STAGE0_MERGE", "False")
    text = replace_assignment(text, "HISTORY_PLIES", str(HISTORY_PLIES))
    text = replace_assignment(text, "USE_STREAM_PRETRAIN_STAGE3", "True" if USE_STREAM_PRETRAIN_STAGE3 else "False")
    path.write_text(text, encoding="utf-8")


COMMON_MODEL_UPDATES = {
    "feature_embed_dim": str(int(profile["feature_embed_dim"])),
    "history_hidden_dim": str(int(profile["history_hidden_dim"])),
    "shared_hidden_dim": str(int(profile["shared_hidden_dim"])),
    "dropout": repr(float(profile["dropout"])),
}


def patch_pretrain_stream_config() -> None:
    path = WORK_ROOT / "scripts" / "script3_pretrain_history_policy_stream.py"
    text = path.read_text(encoding="utf-8")
    updates = {
        "history_plies": str(HISTORY_PLIES),
        "epochs": str(int(profile["pretrain_epochs"])),
        "batch_size": str(int(profile["pretrain_batch_size"])),
        "print_every": str(PRETRAIN_PRINT_EVERY),
        "eval_batch_size": str(int(profile["pretrain_eval_batch_size"])),
        "grad_accum_steps": str(int(profile["pretrain_grad_accum_steps"])),
        "learning_rate": repr(PRETRAIN_LEARNING_RATE),
        "weight_decay": repr(PRETRAIN_WEIGHT_DECAY),
        "strict_target_isolation": "True" if STRICT_TARGET_ISOLATION else "False",
        "shuffle_buffer_size": str(int(profile["pretrain_shuffle_buffer_size"])),
        "scan_progress_every": str(PRETRAIN_SCAN_PROGRESS_EVERY),
        "stream_progress_every": str(PRETRAIN_STREAM_PROGRESS_EVERY),
        "train_eval_max_samples": str(int(profile["pretrain_train_eval_max_samples"])),
        "valid_eval_max_samples": str(int(profile["pretrain_valid_eval_max_samples"])),
    }
    updates.update(COMMON_MODEL_UPDATES)
    for key, value in updates.items():
        text = replace_config_value(text, key, value)
    path.write_text(text, encoding="utf-8")


def patch_pretrain_standard_config() -> None:
    path = WORK_ROOT / "scripts" / "script3_pretrain_history_policy.py"
    text = path.read_text(encoding="utf-8")
    updates = {
        "history_plies": str(HISTORY_PLIES),
        "epochs": str(int(profile["pretrain_epochs"])),
        "batch_size": str(int(profile["pretrain_batch_size"])),
        "print_every": str(PRETRAIN_PRINT_EVERY),
        "eval_batch_size": str(int(profile["pretrain_eval_batch_size"])),
        "grad_accum_steps": str(int(profile["pretrain_grad_accum_steps"])),
        "learning_rate": repr(PRETRAIN_LEARNING_RATE),
        "weight_decay": repr(PRETRAIN_WEIGHT_DECAY),
        "strict_target_isolation": "True" if STRICT_TARGET_ISOLATION else "False",
        "train_eval_max_samples": str(int(profile["pretrain_train_eval_max_samples"])),
        "valid_eval_max_samples": str(int(profile["pretrain_valid_eval_max_samples"])),
    }
    updates.update(COMMON_MODEL_UPDATES)
    for key, value in updates.items():
        text = replace_config_value(text, key, value)
    path.write_text(text, encoding="utf-8")


def patch_finetune_config() -> None:
    path = WORK_ROOT / "scripts" / "script4_finetune_history_policy.py"
    text = path.read_text(encoding="utf-8")
    pretrained_filename = "history_policy_pretrain_stream.pt" if USE_STREAM_PRETRAIN_STAGE3 else "history_policy_pretrain.pt"
    updates = {
        "dataset_tag": json.dumps(DATASET_TAG),
        "history_plies": str(HISTORY_PLIES),
        "pretrained_path": f'models_dir("pretrain_multi") / "{pretrained_filename}"',
        "use_pretrained_encoders": "True" if FINETUNE_USE_PRETRAINED_ENCODERS else "False",
        "epochs": str(int(profile["finetune_epochs"])),
        "batch_size": str(int(profile["finetune_batch_size"])),
        "print_every": str(FINETUNE_PRINT_EVERY),
        "eval_batch_size": str(int(profile["finetune_eval_batch_size"])),
        "grad_accum_steps": str(int(profile["finetune_grad_accum_steps"])),
        "encoder_learning_rate": repr(float(profile["encoder_learning_rate"])),
        "head_learning_rate": repr(float(profile["head_learning_rate"])),
        "weight_decay": repr(float(profile["weight_decay"])),
    }
    updates.update(COMMON_MODEL_UPDATES)
    for key, value in updates.items():
        text = replace_config_value(text, key, value)
    path.write_text(text, encoding="utf-8")


def patch_pretrain_sharded_config() -> None:
    path = WORK_ROOT / "scripts" / "script3_pretrain_history_policy_sharded.py"
    text = path.read_text(encoding="utf-8")
    updates = {
        "history_plies": str(HISTORY_PLIES),
        "epochs": str(int(profile["pretrain_epochs"])),
        "batch_size": str(int(profile["pretrain_batch_size"])),
        "print_every": str(PRETRAIN_PRINT_EVERY),
        "eval_batch_size": str(int(profile["pretrain_eval_batch_size"])),
        "grad_accum_steps": str(int(profile["pretrain_grad_accum_steps"])),
        "learning_rate": repr(PRETRAIN_LEARNING_RATE),
        "weight_decay": repr(PRETRAIN_WEIGHT_DECAY),
        "strict_target_isolation": "True" if STRICT_TARGET_ISOLATION else "False",
        "shuffle_buffer_size": str(int(profile["pretrain_shuffle_buffer_size"])),
        "scan_progress_every": str(PRETRAIN_SCAN_PROGRESS_EVERY),
        "stream_progress_every": str(PRETRAIN_STREAM_PROGRESS_EVERY),
        "train_eval_max_samples": str(int(profile["pretrain_train_eval_max_samples"])),
        "valid_eval_max_samples": str(int(profile["pretrain_valid_eval_max_samples"])),
    }
    updates.update(COMMON_MODEL_UPDATES)
    for key, value in updates.items():
        text = replace_config_value(text, key, value)
    path.write_text(text, encoding="utf-8")


patch_pipeline_config()
patch_pretrain_stream_config()
patch_pretrain_standard_config()
patch_pretrain_sharded_config()
patch_finetune_config()
print("Applied profile:", MODEL_PROFILE)
print(json.dumps(profile, indent=2))
"""
    ),
    code_cell(
        """import importlib
import json
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.chdir(SCRIPTS_DIR)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pipeline_config


def _clear_modules() -> None:
    for name in list(sys.modules):
        if name in {
            "pipeline_config",
            "script1_parse_multi_player_positions",
            "script1_parse_pgn_to_positions",
            "script2_encode_policy_samples",
        }:
            del sys.modules[name]


def preprocess_one_pretrain(src_path: str, out_path: str, history_plies: int) -> tuple[str, int]:
    from pathlib import Path
    import json
    from script1_parse_multi_player_positions import iter_all_player_positions
    from script2_encode_policy_samples import encode_row

    src = Path(src_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in iter_all_player_positions(src, history_plies):
            sample = encode_row(row, history_plies)
            if sample is None:
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\\n")
            count += 1
    return src.name, count


def preprocess_one_target(src_path: str, out_path: str, target_username: str, aliases: list[str], history_plies: int) -> tuple[str, int]:
    from pathlib import Path
    import json
    from script1_parse_pgn_to_positions import iter_target_player_positions
    from script2_encode_policy_samples import encode_row

    src = Path(src_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in iter_target_player_positions(src, target_username, aliases, history_plies):
            sample = encode_row(row, history_plies)
            if sample is None:
                continue
            handle.write(json.dumps(sample, ensure_ascii=False) + "\\n")
            count += 1
    return src.name, count


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def concat_jsonl(parts: list[Path], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as dst:
        for part in sorted(parts):
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        dst.write(line)
                        total += 1
    return total


def run_parallel_jobs(kind: str, files: list[Path], worker_fn, worker_args_builder, tmp_dir: Path, workers: int) -> tuple[list[Path], int]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if not files:
        return [], 0
    parts = []
    total_rows = 0
    max_workers = max(1, min(workers, len(files)))
    print(f"{kind}: files={len(files)} workers={max_workers}")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for idx, path in enumerate(files):
            out_path = tmp_dir / f"part_{idx:04d}_{path.stem}.jsonl"
            parts.append(out_path)
            futures.append(ex.submit(worker_fn, *worker_args_builder(path, out_path)))
        done = 0
        for fut in as_completed(futures):
            name, count = fut.result()
            total_rows += count
            done += 1
            print(f"[{kind}] done {done}/{len(files)} file={name} rows={count}")
    return parts, total_rows


history_plies = HISTORY_PLIES
workers = max(1, PREPROCESS_WORKERS)
processed_pretrain = pipeline_config.processed_dir("pretrain_multi") / "policy_samples.jsonl"
processed_target = pipeline_config.processed_dir(DATASET_TAG) / "policy_samples.jsonl"

pretrain_files = sorted(p for p in PRETRAIN_ROOT.rglob("*.pgn") if p.is_file())
target_root = FINETUNE_ROOT / DATASET_TAG
target_files = sorted(p for p in target_root.rglob("*.pgn") if p.is_file())

if RUN_PRETRAIN and not pretrain_files:
    RUN_PRETRAIN = False
    FINETUNE_USE_PRETRAINED_ENCODERS = False
    print("Auto-disabled pretraining because no pretrain PGNs were found after hydration.")

if RUN_FINETUNE and not target_files:
    raise FileNotFoundError(f"No target PGNs found under {target_root}")

print(f"disk free before preprocessing: {disk_free_gb(WORK_ROOT):.2f} GB")

if RUN_PRETRAIN:
    tmp_pre = pipeline_config.processed_dir("pretrain_multi") / "_parallel_parts"
    if tmp_pre.exists():
        shutil.rmtree(tmp_pre, ignore_errors=True)
    parts, rows = run_parallel_jobs(
        "pretrain",
        pretrain_files,
        preprocess_one_pretrain,
        lambda path, out_path: (str(path), str(out_path), history_plies),
        tmp_pre,
        workers,
    )
    print(f"pretrain shard rows written: {rows} in {tmp_pre}")
    print(f"pretrain shard count: {len(parts)}")
    print(f"disk free after pretrain preprocessing: {disk_free_gb(WORK_ROOT):.2f} GB")

if RUN_FINETUNE:
    tmp_ft = pipeline_config.processed_dir(DATASET_TAG) / "_parallel_parts"
    parts, rows = run_parallel_jobs(
        "finetune",
        target_files,
        preprocess_one_target,
        lambda path, out_path: (str(path), str(out_path), TARGET_USERNAME, TARGET_NAME_ALIASES, history_plies),
        tmp_ft,
        workers,
    )
    total = concat_jsonl(parts, processed_target)
    shutil.rmtree(tmp_ft, ignore_errors=True)
    print(f"finetune rows written: {total} -> {processed_target}")
    print(f"disk free after finetune preprocessing: {disk_free_gb(WORK_ROOT):.2f} GB")
"""
    ),
    code_cell(
        """import importlib
import os
import sys

os.chdir(SCRIPTS_DIR)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

for name in list(sys.modules):
    if name.startswith(("pipeline_config", "script3_", "script4_", "history_policy_lib")):
        del sys.modules[name]

if RUN_PRETRAIN:
    if USE_STREAM_PRETRAIN_STAGE3:
        pretrain_mod = importlib.import_module("script3_pretrain_history_policy_sharded")
    else:
        pretrain_mod = importlib.import_module("script3_pretrain_history_policy")
    pretrain_mod.main()

if RUN_FINETUNE:
    finetune_mod = importlib.import_module("script4_finetune_history_policy")
    finetune_mod.main()
"""
    ),
    code_cell(
        """import json
from pathlib import Path

artifact_paths = [
    WORK_ROOT / "models" / "pretrain_multi" / "history_policy_pretrain_stream.pt",
    WORK_ROOT / "models" / "pretrain_multi" / "history_policy_pretrain_stream_metrics.json",
    WORK_ROOT / "models" / DATASET_TAG / "history_policy.pt",
    WORK_ROOT / "models" / DATASET_TAG / "history_policy_metrics.json",
    WORK_ROOT / "models" / DATASET_TAG / "honest_split_game_ids.json",
]

for path in artifact_paths:
    if path.exists():
        size_mb = path.stat().st_size / (1024 ** 2)
        print(f"{path} | {size_mb:.1f} MB")
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(json.dumps(payload, indent=2)[:4000])
    else:
        print(f"Missing: {path}")
"""
    ),
    code_cell(
        """import shutil
from pathlib import Path

if EXPORT_ARTIFACTS:
    export_root = Path("/kaggle/working") / f"{EXPORT_NAME}_staging"
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    for rel in [Path("models"), Path("outputs")]:
        src = WORK_ROOT / rel
        if src.exists():
            shutil.copytree(src, export_root / rel)

    if EXPORT_PROCESSED:
        processed_src = WORK_ROOT / "data" / "processed"
        if processed_src.exists():
            shutil.copytree(processed_src, export_root / "data" / "processed")

    archive_base = Path("/kaggle/working") / EXPORT_NAME
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=export_root)
    print(f"Created archive: {archive_path}")
else:
    print("EXPORT_ARTIFACTS is disabled.")
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
