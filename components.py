"""HTML and CSS helpers for the movie UI.

Kept free of Streamlit imports so the markup can be tested on its own. All
text coming from the CSV is HTML-escaped before it is placed in the page.
"""

from __future__ import annotations

import html
import re
import zlib
from typing import Iterable, Optional

import pandas as pd

from ai_recommender import split_multi

NO_DESCRIPTION = "No description available."

# Only plain http(s) image URLs without characters that could break out of CSS.
_SAFE_URL = re.compile(r"^https?://[^\s'\"()<>\\]+$")

CSS = """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

.hero {
  padding: 1.6rem 1.6rem 1.4rem; margin-bottom: 1.2rem; border-radius: 18px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(255,77,90,.22), transparent 60%),
              linear-gradient(135deg, #151a26 0%, #0f1320 100%);
  border: 1px solid rgba(255,255,255,.07);
}
.hero h1 { margin: 0 0 .35rem; font-size: clamp(1.6rem, 4.5vw, 2.5rem); line-height: 1.15; }
.hero p  { margin: 0; color: #aab1c5; font-size: clamp(.92rem, 2.4vw, 1.05rem); max-width: 720px; }
.stats { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1rem; }
.stat  { padding: .45rem .9rem; border-radius: 12px; background: rgba(255,255,255,.06);
         border: 1px solid rgba(255,255,255,.08); font-size: .85rem; color: #aab1c5; }
.stat b { display: block; font-size: 1.25rem; color: #fff; line-height: 1.2; }

.movie-grid {
  display: grid; gap: 1rem; margin: .6rem 0 1rem;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
}
.movie-card {
  display: flex; flex-direction: column; overflow: hidden; border-radius: 16px;
  background: #151a26; border: 1px solid rgba(255,255,255,.07);
  transition: transform .15s ease, border-color .15s ease;
}
.movie-card:hover { transform: translateY(-3px); border-color: rgba(255,77,90,.55); }

.poster {
  height: 150px; display: flex; align-items: center; justify-content: center;
  background-size: cover; background-position: center top; background-repeat: no-repeat;
  background-color: #1d2335;
}
.poster.placeholder { flex-direction: column; gap: .2rem; padding: .6rem; text-align: center;
                      font-size: 3.4rem; font-weight: 800; color: rgba(255,255,255,.88); }
.poster.placeholder small { font-size: .8rem; font-weight: 600; letter-spacing: .02em;
                            color: rgba(255,255,255,.7); line-height: 1.2; }

.card-body { padding: .9rem 1rem 1rem; display: flex; flex-direction: column; gap: .55rem; flex: 1; }
.card-title { font-size: 1.12rem; font-weight: 700; margin: 0; line-height: 1.25; }
.card-year  { color: #8b93a9; font-weight: 500; font-size: .9rem; margin-left: .35rem; }
.card-desc  { color: #b9bfd1; font-size: .9rem; line-height: 1.45; margin: 0; }

.chips { display: flex; flex-wrap: wrap; gap: .35rem; }
.chip {
  font-size: .74rem; padding: .18rem .6rem; border-radius: 999px; white-space: nowrap;
  background: rgba(255,255,255,.07); color: #dfe3ef;
}
.chip.genre    { background: rgba(255,77,90,.16);  color: #ff9aa2; }
.chip.mood     { background: rgba(255,184,77,.16); color: #ffcf8f; }
.chip.language { background: rgba(92,170,255,.16); color: #9bcaff; }

.score { margin-top: auto; }
.score-row { display: flex; justify-content: space-between; font-size: .8rem; color: #aab1c5; margin-bottom: .25rem; }
.score-row b { color: #fff; }
.bar { height: 6px; border-radius: 99px; background: rgba(255,255,255,.1); overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, #ff4d5a, #ffb84d); }
.matched { font-size: .76rem; color: #8b93a9; margin-top: .35rem; }
.why { font-size: .8rem; color: #c9cee0; margin: 0; padding: .5rem .65rem; border-radius: 10px;
       background: rgba(255,255,255,.05); border-left: 3px solid #ff4d5a; line-height: 1.4; }
.why b { color: #fff; }

.empty-state {
  text-align: center; padding: 2rem 1rem; border-radius: 16px; margin: .8rem 0;
  background: #151a26; border: 1px dashed rgba(255,255,255,.18); color: #b9bfd1;
}
.empty-state h3 { margin: 0 0 .4rem; color: #fff; }

.detail { display: grid; gap: 1.2rem; grid-template-columns: 220px 1fr; align-items: start; }
.detail .poster { height: 300px; border-radius: 16px; }
.detail h2 { margin: 0 0 .4rem; }
.detail p  { color: #c5cadb; line-height: 1.6; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; }
  .hero { padding: 1.1rem; border-radius: 14px; }
  .movie-grid { grid-template-columns: 1fr; }
  .detail { grid-template-columns: 1fr; }
  .detail .poster { height: 220px; }
}
</style>
"""


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def hero_html(title: str, subtitle: str, stats: Optional[dict] = None) -> str:
    """Hero banner; ``stats`` maps a label to a number, e.g. {"Movies": 34}."""
    stats_html = ""
    if stats:
        items = "".join(
            f'<div class="stat"><b>{_esc(value)}</b>{_esc(label)}</div>'
            for label, value in stats.items()
        )
        stats_html = f'<div class="stats">{items}</div>'
    return f'<div class="hero"><h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p>{stats_html}</div>'


def explanation_text(reasons: object, target: str = "your") -> str:
    """Plain-English explanation built from the engine's top-contributing terms.

    ``target`` is "your" for recommendations and "this movie's" for similar movies.
    """
    if not isinstance(reasons, (list, tuple)) or not reasons:
        return ""
    features = [r for r in reasons if r.startswith(("genre:", "mood:", "language:"))]
    words = [r[len("word: "):] for r in reasons if r.startswith("word: ")]
    parts = []
    if features:
        parts.append(f"matches {target} " + ", ".join(features))
    if words:
        parts.append("shares the words " + ", ".join(words))
    return ("; ".join(parts) + ".").capitalize() if parts else ""


def safe_poster_url(url: object) -> Optional[str]:
    """Return the URL if it is a safe http(s) link, otherwise ``None``."""
    if _is_missing(url):
        return None
    url = str(url).strip()
    return url if _SAFE_URL.match(url) else None


def poster_html(title: str, poster_url: object) -> str:
    """A poster box: the image when a valid URL is given, else a lettered placeholder."""
    url = safe_poster_url(poster_url)
    if url:
        # The gradient sits behind the image, so a broken link still looks fine.
        style = (
            f"background-image:url('{_esc(url)}'),"
            "linear-gradient(135deg,#2a3150,#3a2540)"
        )
        return f'<div class="poster" role="img" aria-label="Poster of {_esc(title)}" style="{style}"></div>'
    return fallback_poster_html(title)


def fallback_poster_html(title: str) -> str:
    """Generated artwork used when no poster image is available.

    The colour comes from the title, so each movie always gets the same colours.
    """
    clean = (title or "?").strip() or "?"
    hue = zlib.crc32(clean.lower().encode("utf-8")) % 360
    gradient = (
        f"linear-gradient(135deg,hsl({hue},55%,30%),hsl({(hue + 55) % 360},60%,16%))"
    )
    return (
        '<div class="poster placeholder" aria-hidden="true" '
        f'style="background-image:{gradient}">{_esc(clean[:1].upper())}'
        f"<small>{_esc(clean)}</small></div>"
    )


def chips_html(row: pd.Series) -> str:
    chips = []
    for css_class, column in (("genre", "genre"), ("mood", "mood"), ("language", "language")):
        for value in split_multi(row.get(column, "")):
            chips.append(f'<span class="chip {css_class}">{_esc(value)}</span>')
    return f'<div class="chips">{"".join(chips)}</div>' if chips else ""


def card_html(row: pd.Series, show_score: bool = True, target: str = "your") -> str:
    """One movie card. Returned as a single line so Markdown never treats it as code."""
    title = _esc(row.get("title", "Untitled"))
    year = row.get("year", "")
    year_html = f'<span class="card-year">{_esc(year)}</span>' if not _is_missing(year) else ""
    description = row.get("description", "")
    description = _esc(description) if not _is_missing(description) else NO_DESCRIPTION

    score_html = ""
    if show_score and "similarity" in row and not _is_missing(row["similarity"]):
        score = max(0.0, min(1.0, float(row["similarity"])))
        matched = row.get("matched", [])
        matched_html = ""
        if isinstance(matched, (list, tuple)) and matched:
            matched_html = f'<div class="matched">Matches your: {_esc(", ".join(matched))}</div>'
        score_html = (
            '<div class="score">'
            f'<div class="score-row"><span>Similarity score</span><b>{score:.2f}</b></div>'
            f'<div class="bar"><i style="width:{score * 100:.0f}%"></i></div>'
            f"{matched_html}</div>"
        )

    why = explanation_text(row.get("reasons", []), target) if show_score else ""
    why_html = f'<p class="why"><b>Why:</b> {_esc(why)}</p>' if why else ""

    return (
        '<div class="movie-card">'
        f"{poster_html(str(row.get('title', '')), row.get('poster_url', ''))}"
        '<div class="card-body">'
        f'<p class="card-title">{title}{year_html}</p>'
        f"{chips_html(row)}"
        f'<p class="card-desc">{description}</p>'
        f"{why_html}"
        f"{score_html}"
        "</div></div>"
    )


def grid_html(movies: pd.DataFrame, show_score: bool = True, target: str = "your") -> str:
    cards = "".join(card_html(row, show_score, target) for _, row in movies.iterrows())
    return f'<div class="movie-grid">{cards}</div>'


def detail_html(row: pd.Series) -> str:
    """Large detail view for a single movie."""
    year = row.get("year", "")
    year_text = f" ({_esc(year)})" if not _is_missing(year) else ""
    description = row.get("description", "")
    description = _esc(description) if not _is_missing(description) else NO_DESCRIPTION
    return (
        '<div class="detail">'
        f"{poster_html(str(row.get('title', '')), row.get('poster_url', ''))}"
        "<div>"
        f"<h2>{_esc(row.get('title', 'Untitled'))}{year_text}</h2>"
        f"{chips_html(row)}"
        f"<p>{description}</p>"
        "</div></div>"
    )


def empty_state_html(title: str, message: str) -> str:
    return f'<div class="empty-state"><h3>{_esc(title)}</h3><p>{_esc(message)}</p></div>'
