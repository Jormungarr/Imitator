#!/usr/bin/env python3
"""Deprecated compatibility entrypoint; use script4_finetune_history_policy.py."""

from __future__ import annotations

import script4_finetune_history_policy as finetune


def main() -> None:
    """Forward to the fine-tuning script for backward compatibility."""
    print("[DEPRECATED] Use script4_finetune_history_policy.py directly.")
    finetune.main()


if __name__ == "__main__":
    main()
