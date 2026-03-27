#!/usr/bin/env python3
"""Bulk-download Chess.com games pages and store PGNs for pretrain or fine-tune."""

from __future__ import annotations

import hashlib
import random
import re
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from playwright.sync_api import BrowserContext, Download, Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

try:
    from playwright._impl._errors import TargetClosedError
except Exception:  # pragma: no cover
    TargetClosedError = Exception

from pipeline_config import DATASET_TAG, player_pgn_dir, pretrain_player_dir, pretrain_player_pgn, raw_pgn


CONFIG = {
    # Default URL derives from DATASET_TAG (override manually if needed).
    "games_url": f"https://www.chess.com/games/{(re.sub(r'[^a-z0-9-]+','-', DATASET_TAG.lower().replace('_','-')).strip('-') or 'unknown-player')}",

    # Storage mode:
    # - pretrain: save under data/raw/pretrain_multi/<player_slug>/ and <player_slug>.pgn
    # - finetune: save under data/raw/finetune_players/<tag>/ and <tag>.pgn
    "storage_mode": "pretrain",
    "finetune_tag": DATASET_TAG,

    # Pagination
    # page_start/page_end are inclusive page numbers.
    # page_end = None means keep scraping until no results.
    "page_start": 21,
    "page_end": 30,
    "page_query_key": "page",

    # Browser behavior
    "headless": False,
    "slow_mo_ms": 100,
    "timeout_ms": 30000,
    "wait_for_manual_login": True,
    "human_like": True,
    "human_pause_min_sec": 0.55,
    "human_pause_max_sec": 3.10,

    # Output naming
    "per_page_subdir": "db_pages",
    "per_page_filename_template": "page_{page:04d}.pgn",
}
SELECTORS = {
    "select_all": [
        "#master-games-check-all",
        "input#master-games-check-all",
        "input[aria-label='Select All Games']",
        "label:has-text('Select all')",
        "button:has-text('Select all')",
        "text=Select all",
        "[aria-label*='Select all' i]",
        "input[type='checkbox'][name*='select' i]",
        "thead input[type='checkbox']",
    ],
    "row_checkbox": [
        "input.master-games-checkbox",
        "input[aria-label='Select the Game']",
        "tbody input[type='checkbox']",
        "[data-cy*='row'] input[type='checkbox']",
        "[class*='row'] input[type='checkbox']",
    ],
    "download_pgn": [
        ".master-games-download-button",
        "button.master-games-download-button",
        "button[aria-label='Download Selected Games']",
        "button:has-text('Download PGN')",
        "a:has-text('Download PGN')",
        "button:has-text('Download Games')",
        "a:has-text('Download Games')",
        "text=Download PGN",
        "text=Download Games",
        "[aria-label*='Download' i]",
        "button:has-text('Download')",
        "a:has-text('Download')",
    ],
    "download_menu_button": [
        ".master-games-download-button",
        "button[aria-label='Download Selected Games']",
        "button:has-text('Download')",
        "a:has-text('Download')",
    ],
    "download_menu_pgn": [
        "button:has-text('PGN')",
        "a:has-text('PGN')",
        "text=PGN",
    ],
    "no_results": [
        "text=Your search did not match any games",
        "text=Please try a new search",
        "text=No games found",
    ],
}


def sanitize_filename(name: str) -> str:
    """Sanitize text for safe filename usage."""
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip("_")


def player_slug_from_games_url(url: str) -> str:
    """Extract player slug from https://www.chess.com/games/<slug>."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "games":
        return sanitize_filename(parts[1]).lower()
    if parts:
        return sanitize_filename(parts[-1]).lower()
    return "unknown_player"


def resolve_output_paths() -> tuple[Path, Path, str]:
    """Resolve per-page output dir and merged output path from storage mode."""
    mode = str(CONFIG.get("storage_mode", "pretrain")).strip().lower()
    games_url = str(CONFIG["games_url"]).strip()
    slug = player_slug_from_games_url(games_url)

    if mode == "pretrain":
        page_dir = pretrain_player_dir(slug) / str(CONFIG["per_page_subdir"])
        merged = pretrain_player_pgn(slug)
        return page_dir, merged, slug

    if mode == "finetune":
        tag = str(CONFIG.get("finetune_tag") or slug).strip()
        if not tag:
            raise ValueError("finetune_tag cannot be empty in finetune mode")
        page_dir = player_pgn_dir(tag) / str(CONFIG["per_page_subdir"])
        merged = raw_pgn(tag)
        return page_dir, merged, tag

    raise ValueError(f"Unsupported storage_mode: {mode}")


def with_query_param(url: str, key: str, value: int) -> str:
    """Return URL with query parameter replaced/added."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query[key] = [str(value)]
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query, doseq=True), parsed.fragment))


def human_pause(page: Page, min_s: Optional[float] = None, max_s: Optional[float] = None) -> None:
    """Pause briefly to behave less like a bot during UI interactions."""
    if not bool(CONFIG.get("human_like", True)):
        return
    lo = float(CONFIG.get("human_pause_min_sec", 0.35) if min_s is None else min_s)
    hi = float(CONFIG.get("human_pause_max_sec", 1.10) if max_s is None else max_s)
    if hi < lo:
        hi = lo
    delay = random.uniform(lo, hi)
    try:
        page.wait_for_timeout(int(delay * 1000))
    except Exception:
        time.sleep(delay)


def try_first_visible(page: Page, selectors: Iterable[str], timeout_ms: int) -> Optional[str]:
    """Return first selector that resolves to a visible element."""
    for selector in selectors:
        try:
            if page.is_closed():
                return None
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=timeout_ms)
            return selector
        except (PlaywrightTimeoutError, TargetClosedError):
            continue
        except Exception:
            continue
    return None


def click_first(page: Page, selectors: Iterable[str], timeout_ms: int, force: bool = False) -> bool:
    """Click first visible selector from a fallback list."""
    selector = try_first_visible(page, selectors, timeout_ms)
    if selector is None:
        return False
    locator = page.locator(selector).first
    human_pause(page)
    try:
        locator.hover()
    except Exception:
        pass
    human_pause(page)
    locator.click(force=force)
    human_pause(page)
    return True


def has_no_results(page: Page) -> bool:
    """Return true if page shows any no-results indicators."""
    if page.is_closed():
        return False
    for selector in SELECTORS["no_results"]:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0 and locator.is_visible():
                return True
        except Exception:
            continue
    return False


def checked_checkbox_count(page: Page) -> int:
    """Return count of checked game-selection checkboxes."""
    if page.is_closed():
        return 0

    max_count = 0
    for selector in [
        "input[type='checkbox']:checked",
        "input[aria-checked='true']",
        "input.master-games-checkbox:checked",
    ]:
        try:
            max_count = max(max_count, page.locator(selector).count())
        except Exception:
            continue
    return max_count


def ensure_selection_before_download(page: Page) -> bool:
    """Try robustly selecting page games before download."""
    before = checked_checkbox_count(page)
    if click_first(page, SELECTORS["select_all"], 5000, force=True):
        human_pause(page, 0.25, 0.8)
    after = checked_checkbox_count(page)
    if after > before:
        return True

    row_selector = try_first_visible(page, SELECTORS["row_checkbox"], 4000)
    if row_selector is not None:
        try:
            page.locator(row_selector).first.click(force=True)
            human_pause(page, 0.25, 0.8)
        except Exception:
            pass

    before_retry = checked_checkbox_count(page)
    click_first(page, SELECTORS["select_all"], 4000, force=True)
    human_pause(page, 0.25, 0.8)
    after_retry = checked_checkbox_count(page)
    return after_retry > before_retry or after_retry > 0


def save_download(download: Download, out_path: Path) -> None:
    """Persist one Playwright download to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    download.save_as(str(out_path))


def dump_debug_page(page: Page, out_dir: Path, page_idx: int) -> None:
    """Write debug screenshot/html for a failed page."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"debug_page_{page_idx:04d}.png"
    html = out_dir / f"debug_page_{page_idx:04d}.html"
    try:
        page.screenshot(path=str(png), full_page=True)
    except Exception:
        pass
    try:
        html.write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


def download_current_page_pgn(page: Page, page_idx: int, out_dir: Path, timeout_ms: int) -> Optional[Path]:
    """Download PGN for all selected games on current page."""
    if page.is_closed():
        return None
    if not ensure_selection_before_download(page):
        print(f"[PAGE {page_idx}] Could not confirm game selection; skip.")
        return None

    download_obj: Optional[Download] = None

    try:
        with page.expect_download(timeout=timeout_ms) as download_info:
            click_first(page, SELECTORS["download_pgn"], timeout_ms, force=True)
        download_obj = download_info.value
    except (PlaywrightTimeoutError, TargetClosedError):
        download_obj = None

    if download_obj is None:
        try:
            opened = click_first(page, SELECTORS["download_menu_button"], 5000, force=True)
            if not opened:
                return None
            human_pause(page, 0.25, 0.8)
            with page.expect_download(timeout=timeout_ms) as download_info:
                ok = click_first(page, SELECTORS["download_menu_pgn"], 5000, force=True)
                if not ok:
                    return None
            download_obj = download_info.value
        except (PlaywrightTimeoutError, TargetClosedError):
            return None

    if download_obj is None:
        return None

    out_name = str(CONFIG["per_page_filename_template"]).format(page=page_idx)
    out_path = out_dir / sanitize_filename(out_name)
    save_download(download_obj, out_path)
    return out_path


def merge_unique_pgn_chunks(page_files: list[Path], merged_path: Path) -> None:
    """Merge page PGNs with chunk-level deduplication."""
    seen_hashes = set()
    chunks = []

    for file_path in page_files:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        sha = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        if sha in seen_hashes:
            continue
        seen_hashes.add(sha)
        chunks.append(text)

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged = "\n\n".join(chunks).strip()
    merged_path.write_text((merged + "\n") if merged else "", encoding="utf-8")


def get_live_page(context: BrowserContext, current_page: Optional[Page], timeout_ms: int) -> Page:
    """Get a usable page from context when popups/closures happen."""
    if current_page is not None and not current_page.is_closed():
        return current_page
    for page in reversed(context.pages):
        if not page.is_closed():
            page.set_default_timeout(timeout_ms)
            return page
    page = context.new_page()
    page.set_default_timeout(timeout_ms)
    return page


def main() -> None:
    """Iterate games pages, download per-page PGNs, and merge them."""
    base_url = str(CONFIG["games_url"]).strip()
    if not base_url:
        raise ValueError("CONFIG['games_url'] cannot be empty")

    out_dir, merged_output_path, storage_label = resolve_output_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    timeout_ms = int(CONFIG["timeout_ms"])
    page_start = int(CONFIG.get("page_start", 1))
    page_end_raw = CONFIG.get("page_end", None)
    page_end = None if page_end_raw in (None, "", "none", "None") else int(page_end_raw)
    page_key = str(CONFIG["page_query_key"]).strip() or "page"

    print(f"Storage mode: {CONFIG['storage_mode']}")
    print(f"Storage key: {storage_label}")
    print(f"Page PGN dir: {out_dir}")
    print(f"Merged PGN: {merged_output_path}")
    print(f"Page range: start={page_start}, end={page_end}")

    downloaded_files: list[Path] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=bool(CONFIG["headless"]), slow_mo=int(CONFIG["slow_mo_ms"]))
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(timeout_ms)

        first_url = with_query_param(base_url, page_key, page_start)
        print(f"Opening: {first_url}")
        page.goto(first_url, wait_until="domcontentloaded")

        if bool(CONFIG["wait_for_manual_login"]):
            input("Complete login/captcha in browser if needed, then press Enter to continue...")

        page_idx = page_start
        while True:
            url = with_query_param(base_url, page_key, page_idx)
            print(f"\n[PAGE {page_idx}] {url}")

            page = get_live_page(context, page, timeout_ms)
            try:
                page.goto(url, wait_until="domcontentloaded")
            except TargetClosedError:
                page = get_live_page(context, None, timeout_ms)
                page.goto(url, wait_until="domcontentloaded")

            human_pause(page, 0.9, 1.8)
            page = get_live_page(context, page, timeout_ms)

            if has_no_results(page):
                if page_idx == page_start:
                    raise RuntimeError("No games on page 1. Check URL/login state.")
                print("No more results; stop.")
                break

            file_path = download_current_page_pgn(page, page_idx, out_dir, timeout_ms)
            if file_path is None:
                if page_idx == page_start:
                    dump_debug_page(page, out_dir, page_idx)
                    raise RuntimeError(
                        "Failed to download page 1. Debug files saved under "
                        f"{out_dir}."
                    )
                print("Download unavailable on this page; stop.")
                break

            print(f"Saved: {file_path.name}")
            downloaded_files.append(file_path)

            if page_end is not None and page_idx >= page_end:
                print("Reached configured page_end; stop.")
                break

            page_idx += 1

        if not downloaded_files:
            raise RuntimeError("No page PGNs downloaded.")

        merge_unique_pgn_chunks(downloaded_files, merged_output_path)
        print("\nDone.")
        print(f"Page files downloaded: {len(downloaded_files)}")
        print(f"Merged PGN: {merged_output_path}")

        context.close()
        browser.close()


if __name__ == "__main__":
    main()




