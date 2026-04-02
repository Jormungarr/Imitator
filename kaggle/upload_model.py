#!/usr/bin/env python3
"""Upload a trained model directory as a Kaggle Model using kagglehub.

Usage:
    python kaggle/upload_model.py

Edit the CONFIG block below before running.
"""

from __future__ import annotations

from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "owner_slug": "chess",
    "model_slug": "imitator-history-policy",
    "framework": "pyTorch",
    "variation_slug": "default",
    "local_model_dir": PROJECT_ROOT / "models" / "Bobby_Fischer",
    "version_notes": "Update 2026-03-29",
}


def main() -> None:
    """Upload a local model directory as a Kaggle Model version."""
    local_model_dir = Path(CONFIG["local_model_dir"]).resolve()
    if not local_model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {local_model_dir}")
    if not local_model_dir.is_dir():
        raise NotADirectoryError(f"local_model_dir must be a directory: {local_model_dir}")

    handle = (
        f"{CONFIG['owner_slug']}/{CONFIG['model_slug']}/"
        f"{CONFIG['framework']}/{CONFIG['variation_slug']}"
    )

    print(f"Uploading model from: {local_model_dir}")
    print(f"Handle: {handle}")

    kagglehub.model_upload(
        handle=handle,
        local_model_dir=str(local_model_dir),
        version_notes=str(CONFIG["version_notes"]),
    )

    print("Model upload completed.")


if __name__ == "__main__":
    main()

