"""Optional poster lookup using The Movie Database (TMDB) API.

Real posters are shown in this order of preference:
1. the ``poster_url`` column in movies.csv (if it holds a valid http/https link);
2. a poster found through TMDB, but only when a free TMDB API key is configured;
3. a generated fallback artwork (always available, no internet needed).

Every failure (no key, no internet, movie not found, bad response) quietly returns
``None`` so the app keeps working and shows the fallback artwork.

This module does not import Streamlit, so it can be tested on its own.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Optional

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
REQUEST_TIMEOUT_SECONDS = 5
API_KEY_NAME = "TMDB_API_KEY"

TMDB_NOTICE = "This product uses the TMDB API but is not endorsed or certified by TMDB."


def api_key_from_environment() -> Optional[str]:
    """Read the key from the environment (Streamlit secrets are handled in app.py)."""
    key = os.environ.get(API_KEY_NAME, "").strip()
    return key or None


def fetch_tmdb_poster(title: str, year: object, api_key: Optional[str]) -> Optional[str]:
    """Return a poster image URL for a movie, or ``None`` if none can be found."""
    if not api_key or not str(title).strip():
        return None

    params = {"api_key": api_key, "query": str(title).strip(), "include_adult": "false"}
    year_text = str(year).strip()
    if year_text.isdigit() and len(year_text) == 4:
        params["year"] = year_text

    url = f"{TMDB_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        for result in payload.get("results", []):
            poster_path = result.get("poster_path")
            if poster_path and isinstance(poster_path, str) and poster_path.startswith("/"):
                return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception:  # noqa: BLE001 - any failure means "use the fallback artwork"
        return None
    return None
