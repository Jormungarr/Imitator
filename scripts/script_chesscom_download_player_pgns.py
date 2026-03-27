#!/usr/bin/env python3
"""Download Chess.com monthly PGN archives for one player."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from pipeline_config import DATASET_TAG, raw_dir, raw_pgn


ARCHIVE_URL_RE = re.compile(r"/games/(\d{4})/(\d{2})/?$")

CONFIG = {
    "username": "hikaru",
    "dataset_tag": DATASET_TAG,
    "monthly_output_dir": raw_dir() / DATASET_TAG,
    "monthly_filename_template": "{yyyy}_{mm}.pgn",
    "write_merged_pgn": True,
    "merged_output_path": raw_pgn(DATASET_TAG),
    "start_ym": None,
    "end_ym": None,
    "overwrite_monthly": False,
    "sleep_seconds_between_requests": 0.25,
    "timeout_seconds": 30,
    "max_retries": 4,
    "user_agent": "imitator-pgn-downloader/1.0 (contact: your_email@example.com)",
}


@dataclass(frozen=True)
class ArchiveItem:
    url: str
    year: int
    month: int

    @property
    def ym_key(self) -> Tuple[int, int]:
        return (self.year, self.month)


def _request_bytes(url: str) -> bytes:
    timeout = int(CONFIG["timeout_seconds"])
    max_retries = int(CONFIG["max_retries"])
    ua = str(CONFIG["user_agent"])

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                time.sleep((2 ** attempt) * 0.8)
                continue
            raise
        except urllib.error.URLError:
            if attempt < max_retries:
                time.sleep((2 ** attempt) * 0.8)
                continue
            raise

    raise RuntimeError(f"Request failed after retries: {url}")


def fetch_json(url: str) -> Any:
    return json.loads(_request_bytes(url).decode("utf-8"))


def fetch_text(url: str) -> str:
    return _request_bytes(url).decode("utf-8", errors="replace")


def parse_archive_item(url: str) -> Optional[ArchiveItem]:
    match = ARCHIVE_URL_RE.search(url.strip())
    if not match:
        return None
    return ArchiveItem(url=url.rstrip("/"), year=int(match.group(1)), month=int(match.group(2)))


def in_range(item: ArchiveItem, start_ym: Optional[Tuple[int, int]], end_ym: Optional[Tuple[int, int]]) -> bool:
    if start_ym is not None and item.ym_key < start_ym:
        return False
    if end_ym is not None and item.ym_key > end_ym:
        return False
    return True


def archive_list_url(username: str) -> str:
    return f"https://api.chess.com/pub/player/{username}/games/archives"


def monthly_pgn_url(archive_url: str) -> str:
    return archive_url.rstrip("/") + "/pgn"


def main() -> None:
    username = str(CONFIG["username"]).strip()
    if not username:
        raise ValueError("CONFIG['username'] cannot be empty")

    monthly_dir = Path(CONFIG["monthly_output_dir"])
    monthly_dir.mkdir(parents=True, exist_ok=True)

    archives_json = fetch_json(archive_list_url(username))
    archive_urls = archives_json.get("archives", [])
    if not archive_urls:
        raise ValueError(f"No archives found for player: {username}")

    start_ym = CONFIG.get("start_ym")
    end_ym = CONFIG.get("end_ym")
    if start_ym is not None:
        start_ym = (int(start_ym[0]), int(start_ym[1]))
    if end_ym is not None:
        end_ym = (int(end_ym[0]), int(end_ym[1]))

    archives: List[ArchiveItem] = []
    for url in archive_urls:
        item = parse_archive_item(str(url))
        if item and in_range(item, start_ym, end_ym):
            archives.append(item)
    archives.sort(key=lambda x: x.ym_key)

    merged_chunks: List[str] = []
    overwrite = bool(CONFIG["overwrite_monthly"])

    for item in archives:
        out_name = str(CONFIG["monthly_filename_template"]).format(yyyy=item.year, mm=f"{item.month:02d}")
        out_file = monthly_dir / out_name

        if out_file.exists() and not overwrite:
            text = out_file.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                merged_chunks.append(text)
            continue

        text = fetch_text(monthly_pgn_url(item.url)).strip()
        if text:
            out_file.write_text(text + "\n", encoding="utf-8")
            merged_chunks.append(text)

        sleep_s = float(CONFIG["sleep_seconds_between_requests"])
        if sleep_s > 0:
            time.sleep(sleep_s)

    if bool(CONFIG["write_merged_pgn"]):
        merged_output = Path(CONFIG["merged_output_path"])
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        merged_output.write_text("\n\n".join(merged_chunks).strip() + "\n", encoding="utf-8")
        print(f"Wrote merged PGN: {merged_output}")


if __name__ == "__main__":
    main()
