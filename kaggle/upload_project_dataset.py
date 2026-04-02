#!/usr/bin/env python3
"""Upload this project root as a Kaggle dataset using kagglehub.

Usage:
    python kaggle/upload_project_dataset.py

Edit the CONFIG block below before running.
"""

from __future__ import annotations

from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "handle": "jormungand/imitator-project",
    "local_dataset_dir": PROJECT_ROOT,
    "version_notes": "Project snapshot for Kaggle notebook usage",
    "ignore_patterns": [
        ".git/",
        "__pycache__/",
        "*.pyc",
        ".ipynb_checkpoints/",
        ".venv/",
        "venv/",
        "env/",
        ".playwright_chesscom_profile/",
        "data/processed/",
        "models/",
        "outputs/",
    ],
}


def main() -> None:
    """Upload the project directory as a Kaggle dataset."""
    local_dataset_dir = Path(CONFIG["local_dataset_dir"]).resolve()
    if not local_dataset_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {local_dataset_dir}")

    print(f"Uploading dataset from: {local_dataset_dir}")
    print(f"Handle: {CONFIG['handle']}")

    kagglehub.dataset_upload(
        handle=str(CONFIG["handle"]),
        local_dataset_dir=str(local_dataset_dir),
        version_notes=str(CONFIG["version_notes"]),
        ignore_patterns=list(CONFIG["ignore_patterns"]),
    )

    print("Dataset upload completed.")


if __name__ == "__main__":
    main()

