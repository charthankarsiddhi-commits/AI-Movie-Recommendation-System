"""Content-based movie recommendation engine.

The engine turns every movie into a small text "profile" made of its genre,
mood, language and description, converts those profiles into TF-IDF vectors,
and ranks movies by cosine similarity to the user's preferences.

This module has no Streamlit dependency, so it can be imported and tested on
its own.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Relative to this file, so it works locally and on Streamlit Community Cloud.
DATA_PATH = Path(__file__).resolve().parent / "movies.csv"

REQUIRED_COLUMNS = ["title", "genre", "mood", "language", "description"]
OPTIONAL_COLUMNS = ["year", "poster_url"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: Sentinel used by the UI for "no preference".
ANY = "Any"

#: Several genres or moods can be stored in one CSV cell, e.g. "Action|Drama".
MULTI_VALUE_SEPARATORS = r"[|;]"

#: Feature weights. A feature is repeated this many times in the movie profile
#: (and in the query), so genre counts more than language, and language and mood
#: count more than any single word of the description.
GENRE_WEIGHT = 3
MOOD_WEIGHT = 2
LANGUAGE_WEIGHT = 2

MAX_KEYWORD_LENGTH = 120
DEFAULT_MIN_SCORE = 0.05
DEFAULT_TOP_N = 5


class InputValidationError(ValueError):
    """Raised when the user's preferences are invalid."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def split_multi(value: object) -> List[str]:
    """Split a cell such as ``"Action|Drama"`` into clean, de-duplicated parts."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parts = [p.strip() for p in re.split(MULTI_VALUE_SEPARATORS, str(value))]
    seen, result = set(), []
    for part in parts:
        if part and part.lower() not in seen:
            seen.add(part.lower())
            result.append(part)
    return result


def _feature_token(prefix: str, value: str) -> str:
    """Turn ``("genre", "Science Fiction")`` into the single token ``genre_science_fiction``."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return f"{prefix}_{slug}" if slug else ""


def _feature_tokens(prefix: str, values: Iterable[str], weight: int) -> List[str]:
    tokens = [_feature_token(prefix, v) for v in values]
    return [t for t in tokens if t] * weight


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_keywords(text: Optional[str]) -> str:
    """Keep letters, digits and spaces from optional free-text keywords."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text))).strip()


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_movies(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load and clean the movie CSV.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty, unreadable, lacks required columns
            or contains no usable rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Movie dataset not found at '{path.name}'. "
            "Make sure data/movies.csv is committed to the repository."
        )
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The movie dataset file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"The movie dataset could not be read: {exc}") from exc

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            "The movie dataset is missing required column(s): " + ", ".join(missing)
        )
    for column in OPTIONAL_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""

    df = raw[ALL_COLUMNS].copy()
    for column in ALL_COLUMNS:
        df[column] = df[column].map(_clean_text)

    # Drop rows without a title and duplicate titles (case-insensitive).
    df = df[df["title"] != ""]
    df = df.loc[~df["title"].str.lower().duplicated(keep="first")]

    # A year must be 4 digits; anything else is treated as missing.
    df["year"] = df["year"].where(df["year"].str.fullmatch(r"\d{4}"), "")

    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("The movie dataset has no usable rows.")
    return df


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Preferences:
    """Validated user preferences."""

    genre: str = ANY
    mood: str = ANY
    language: str = ANY
    keywords: str = ""


class MovieRecommender:
    """Content-based recommender using TF-IDF and cosine similarity."""

    def __init__(self, movies: pd.DataFrame):
        if movies is None or movies.empty:
            raise ValueError("Cannot build a recommender from an empty dataset.")
        self.movies = movies.reset_index(drop=True).copy()

        documents = [self._movie_document(row) for row in self.movies.itertuples()]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            sublinear_tf=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        )
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        except ValueError as exc:  # e.g. every document is empty
            raise ValueError(
                "The dataset has no usable genre, mood, language or description text."
            ) from exc

        self.genres = self._unique_values("genre")
        self.moods = self._unique_values("mood")
        self.languages = self._unique_values("language")
        self._title_index = {t.lower(): i for i, t in enumerate(self.movies["title"])}

    # -- feature construction ------------------------------------------------ #

    @staticmethod
    def _movie_document(row) -> str:
        """The text profile of one movie that gets vectorised."""
        parts: List[str] = []
        parts += _feature_tokens("genre", split_multi(row.genre), GENRE_WEIGHT)
        parts += _feature_tokens("mood", split_multi(row.mood), MOOD_WEIGHT)
        parts += _feature_tokens("language", split_multi(row.language), LANGUAGE_WEIGHT)
        parts.append(row.description)
        return " ".join(parts)

    @staticmethod
    def _query_document(prefs: Preferences) -> str:
        """The text profile of the user's preferences, built the same way."""
        parts: List[str] = []
        if prefs.genre != ANY:
            parts += _feature_tokens("genre", [prefs.genre], GENRE_WEIGHT)
        if prefs.mood != ANY:
            parts += _feature_tokens("mood", [prefs.mood], MOOD_WEIGHT)
        if prefs.language != ANY:
            parts += _feature_tokens("language", [prefs.language], LANGUAGE_WEIGHT)
        if prefs.keywords:
            parts.append(prefs.keywords)
        return " ".join(parts)

    def _unique_values(self, column: str) -> List[str]:
        values = {}
        for cell in self.movies[column]:
            for value in split_multi(cell):
                values.setdefault(value.lower(), value)
        return sorted(values.values(), key=str.lower)

    # -- validation ---------------------------------------------------------- #

    def validate(
        self,
        genre: Optional[str] = ANY,
        mood: Optional[str] = ANY,
        language: Optional[str] = ANY,
        keywords: Optional[str] = "",
    ) -> Preferences:
        """Check user input and return cleaned :class:`Preferences`.

        Raises:
            InputValidationError: if a value is unknown, the keywords are too
                long, or no preference was given at all.
        """

        def check(label: str, value: Optional[str], options: Sequence[str]) -> str:
            value = _clean_text(value) or ANY
            if value == ANY:
                return ANY
            lookup = {o.lower(): o for o in options}
            if value.lower() not in lookup:
                raise InputValidationError(
                    f"'{value}' is not a valid {label}. "
                    f"Choose one of: {', '.join(options)}."
                )
            return lookup[value.lower()]

        keywords = _clean_text(keywords)
        if len(keywords) > MAX_KEYWORD_LENGTH:
            raise InputValidationError(
                f"Keywords are too long (max {MAX_KEYWORD_LENGTH} characters)."
            )

        prefs = Preferences(
            genre=check("genre", genre, self.genres),
            mood=check("mood", mood, self.moods),
            language=check("language", language, self.languages),
            keywords=_clean_keywords(keywords),
        )
        if prefs.genre == ANY and prefs.mood == ANY and prefs.language == ANY and not prefs.keywords:
            raise InputValidationError(
                "Please choose at least one genre, mood or language "
                "(or type a keyword) to get recommendations."
            )
        return prefs

    # -- recommendations ----------------------------------------------------- #

    def recommend(
        self,
        genre: Optional[str] = ANY,
        mood: Optional[str] = ANY,
        language: Optional[str] = ANY,
        keywords: Optional[str] = "",
        top_n: int = DEFAULT_TOP_N,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> pd.DataFrame:
        """Rank movies by cosine similarity to the user's preferences.

        Returns a DataFrame with the movie columns plus ``similarity`` (cosine
        similarity between the preference vector and the movie vector, from
        0 to 1) and ``matched`` (which chosen preferences the movie satisfies).
        Movies scoring below ``min_score`` are left out, so the result may be
        empty.
        """
        prefs = self.validate(genre, mood, language, keywords)
        if not isinstance(top_n, int) or top_n < 1:
            raise InputValidationError("top_n must be a positive whole number.")
        if not 0 <= float(min_score) <= 1:
            raise InputValidationError("min_score must be between 0 and 1.")

        query_vector = self.vectorizer.transform([self._query_document(prefs)])
        if query_vector.nnz == 0:
            # None of the query words exist in the dataset vocabulary.
            return self._empty_result()

        scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
        return self._rank(scores, top_n, min_score, prefs, query_vector)

    def similar_to(
        self, title: str, top_n: int = DEFAULT_TOP_N, min_score: float = 0.0
    ) -> pd.DataFrame:
        """Movies most similar to an existing movie (the movie itself is excluded)."""
        index = self._title_index.get(_clean_text(title).lower())
        if index is None:
            raise InputValidationError(f"Movie '{title}' is not in the catalogue.")
        scores = cosine_similarity(self.tfidf_matrix[index], self.tfidf_matrix).ravel()
        scores[index] = -1.0  # never recommend the movie itself
        return self._rank(scores, top_n, min_score, None, self.tfidf_matrix[index])

    def _rank(
        self,
        scores: np.ndarray,
        top_n: int,
        min_score: float,
        prefs: Optional[Preferences],
        query_vector,
    ) -> pd.DataFrame:
        result = self.movies.copy()
        result["similarity"] = np.round(scores, 4)
        result["matched"] = [self._matched(row, prefs) for row in self.movies.itertuples()]
        result = result[result["similarity"] > 0]
        result = result[result["similarity"] >= min_score]
        # Highest score first; ties broken alphabetically for stable output.
        result = result.sort_values(["similarity", "title"], ascending=[False, True])
        result = result.head(top_n).copy()
        result["reasons"] = [self._explain(query_vector, i) for i in result.index]
        return result.reset_index(drop=True)

    def _empty_result(self) -> pd.DataFrame:
        return pd.DataFrame(columns=ALL_COLUMNS + ["similarity", "matched", "reasons"])

    # -- explanations -------------------------------------------------------- #

    @staticmethod
    def _readable(term: str) -> str:
        """``genre_science_fiction`` -> ``genre: science fiction``; plain words get ``word:``."""
        for prefix in ("genre", "mood", "language"):
            if term.startswith(prefix + "_"):
                return f"{prefix}: {term[len(prefix) + 1:].replace('_', ' ')}"
        return f"word: {term}"

    def _explain(self, query_vector, movie_index: int, top_k: int = 4) -> List[str]:
        """Terms that contributed most to the similarity between a query and a movie.

        Cosine similarity is a sum over terms of (query weight x movie weight);
        the terms with the biggest products are the best explanation of the score.
        """
        shared = query_vector.multiply(self.tfidf_matrix[movie_index]).tocoo()
        if shared.nnz == 0:
            return []
        names = self.vectorizer.get_feature_names_out()
        ranked = sorted(zip(shared.col, shared.data), key=lambda pair: -pair[1])[:top_k]
        return [self._readable(str(names[col])) for col, _ in ranked]

    def top_terms(self, title: str, n: int = 8) -> pd.DataFrame:
        """The highest-weighted TF-IDF terms of one movie (for the 'How it works' page)."""
        index = self._title_index.get(_clean_text(title).lower())
        if index is None:
            raise InputValidationError(f"Movie '{title}' is not in the catalogue.")
        row = self.tfidf_matrix[index].tocoo()
        names = self.vectorizer.get_feature_names_out()
        ranked = sorted(zip(row.col, row.data), key=lambda pair: -pair[1])[:n]
        return pd.DataFrame(
            {
                "Term": [self._readable(str(names[c])) for c, _ in ranked],
                "TF-IDF weight": [round(float(w), 3) for _, w in ranked],
            }
        )

    def stats(self) -> dict:
        """Headline numbers for the UI."""
        return {
            "movies": len(self.movies),
            "genres": len(self.genres),
            "moods": len(self.moods),
            "languages": len(self.languages),
            "vocabulary": int(self.tfidf_matrix.shape[1]),
        }

    @staticmethod
    def _matched(row, prefs: Optional[Preferences]) -> List[str]:
        """Which of the user's chosen preferences this movie satisfies."""
        if prefs is None:
            return []
        matched = []
        checks = (
            ("genre", prefs.genre, row.genre),
            ("mood", prefs.mood, row.mood),
            ("language", prefs.language, row.language),
        )
        for label, wanted, actual in checks:
            if wanted != ANY and wanted.lower() in [v.lower() for v in split_multi(actual)]:
                matched.append(label)
        return matched

    # -- lookup -------------------------------------------------------------- #

    def search(self, query: Optional[str]) -> pd.DataFrame:
        """Find movies by title: substring matches first, then close spellings."""
        query = _clean_text(query).lower()
        if not query:
            return self.movies.copy()

        titles = self.movies["title"].str.lower()
        exact_or_partial = self.movies[titles.str.contains(re.escape(query), regex=True)]
        found = list(exact_or_partial.index)

        if not found:
            close = difflib.get_close_matches(query, list(titles), n=5, cutoff=0.6)
            found = [self._title_index[c] for c in close if c in self._title_index]

        return self.movies.loc[found].reset_index(drop=True)

    def get_movie(self, title: str) -> Optional[pd.Series]:
        """Return one movie by exact (case-insensitive) title, or ``None``."""
        index = self._title_index.get(_clean_text(title).lower())
        return None if index is None else self.movies.iloc[index]
