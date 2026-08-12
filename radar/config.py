"""Paths and settings. Everything file-based lives under a single cache root."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")

#: Root of the on-disk cache. Overridable so a deployed app can point at a
#: read-only bundled snapshot instead of a writable scratch dir.
CACHE_DIR = Path(os.environ.get("RADAR_CACHE_DIR") or REPO_ROOT / "data" / "cache")

#: Raw per-ticker OHLC+adjusted price history, one parquet per ticker.
EOD_DIR = CACHE_DIR / "eod"

#: Precomputed rolling-window artifacts consumed by the app (phase 2).
ARTIFACT_DIR = CACHE_DIR / "artifacts"

TIINGO_BASE_URL = "https://api.tiingo.com"

#: Tiingo's free tier is limited per hour/day. We stay well under by caching
#: aggressively and only ever fetching the missing tail of a series.
REQUEST_TIMEOUT = 30.0
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0
INTER_REQUEST_SLEEP = 0.15


class ConfigError(RuntimeError):
    pass


def tiingo_api_key() -> str:
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "TIINGO_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free at https://www.tiingo.com/account/api/token)."
        )
    return key


def ensure_dirs() -> None:
    EOD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
