"""AI Movie Recommendation System - Streamlit user interface.

All recommendation logic lives in ai_recommender.py; this file only draws the page.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from ai_recommender import (
    ANY,
    DATA_PATH,
    MAX_KEYWORD_LENGTH,
    InputValidationError,
    MovieRecommender,
    load_movies,
    split_multi,
)
from components import (
    CSS,
    detail_html,
    empty_state_html,
    grid_html,
    hero_html,
    safe_poster_url,
)
from posters import API_KEY_NAME, TMDB_NOTICE, api_key_from_environment, fetch_tmdb_poster

st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data, model and posters (cached)
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading movies and training the TF-IDF model...")
def build_recommender(file_mtime: float) -> MovieRecommender:
    """``file_mtime`` is only a cache key: editing movies.csv rebuilds the model."""
    return MovieRecommender(load_movies(DATA_PATH))


def get_recommender() -> MovieRecommender:
    try:
        mtime = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
        return build_recommender(mtime)
    except (FileNotFoundError, ValueError) as exc:
        st.error(
            f"Could not load the movie dataset. {exc} "
            "Check that movies.csv is in the same folder as app.py."
        )
        st.stop()


def tmdb_api_key() -> Optional[str]:
    """TMDB key from Streamlit secrets or an environment variable (optional)."""
    try:
        key = st.secrets.get(API_KEY_NAME)
    except Exception:  # no secrets file configured
        key = None
    return (str(key).strip() if key else "") or api_key_from_environment()


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_poster(title: str, year: str, key: str) -> Optional[str]:
    return fetch_tmdb_poster(title, year, key)


def with_posters(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing poster links from TMDB when a key is set; otherwise leave as is.

    Movies that still have no poster get generated fallback artwork when drawn.
    """
    key = tmdb_api_key()
    if not key or frame.empty:
        return frame
    frame = frame.copy()
    with st.spinner("Fetching posters..."):
        for i, row in frame.iterrows():
            if safe_poster_url(row.get("poster_url")) is None:
                frame.at[i, "poster_url"] = cached_poster(row["title"], row.get("year", ""), key) or ""
    return frame


recommender = get_recommender()
movies = recommender.movies
catalogue_stats = recommender.stats()

# --------------------------------------------------------------------------- #
# Preference state and reset
# --------------------------------------------------------------------------- #

TOP_N_MAX = max(2, min(10, len(movies)))
DEFAULTS = {
    "pref_genre": ANY,
    "pref_mood": ANY,
    "pref_language": ANY,
    "pref_keywords": "",
    "pref_top_n": min(5, TOP_N_MAX),
}
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
st.session_state.setdefault("results", None)


def reset_preferences() -> None:
    """Callback for the Clear button: restore every input and drop old results."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["results"] = None


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #

st.markdown(
    hero_html(
        "🎬 AI Movie Recommendation System",
        "Choose a genre, mood and language and get movies whose profiles are most similar "
        "to your taste. A content-based AI model converts every movie into a TF-IDF vector "
        "and ranks the catalogue by cosine similarity.",
        {
            "Movies": catalogue_stats["movies"],
            "Genres": catalogue_stats["genres"],
            "Moods": catalogue_stats["moods"],
            "Languages": catalogue_stats["languages"],
        },
    ),
    unsafe_allow_html=True,
)

tab_recommend, tab_search, tab_catalogue, tab_how = st.tabs(
    ["🎯 Recommend", "🔎 Search & details", "📚 Catalogue", "🧠 How it works"]
)

# ----------------------------- Recommend tab ------------------------------- #
with tab_recommend:
    col_genre, col_mood, col_language = st.columns(3)
    genre = col_genre.selectbox("Genre", [ANY] + recommender.genres, key="pref_genre")
    mood = col_mood.selectbox("Mood", [ANY] + recommender.moods, key="pref_mood")
    language = col_language.selectbox(
        "Language", [ANY] + recommender.languages, key="pref_language"
    )

    col_keywords, col_count = st.columns([3, 1])
    keywords = col_keywords.text_input(
        "Keywords (optional)",
        key="pref_keywords",
        max_chars=MAX_KEYWORD_LENGTH,
        placeholder="e.g. friendship, space, heist",
        help="Words are matched against movie descriptions.",
    )
    top_n = col_count.slider(
        "How many movies?", min_value=1, max_value=TOP_N_MAX, key="pref_top_n"
    )

    col_go, col_clear, _spacer = st.columns([1, 1, 2])
    go = col_go.button("🎬 Recommend Movies", type="primary")
    col_clear.button("↺ Clear preferences", on_click=reset_preferences)

    if go:
        try:
            with st.spinner("Finding the best matches..."):
                prefs = recommender.validate(genre, mood, language, keywords)
                found = recommender.recommend(
                    prefs.genre, prefs.mood, prefs.language, prefs.keywords, top_n=int(top_n)
                )
                found = with_posters(found)
            st.session_state["results"] = {"prefs": prefs, "movies": found}
        except InputValidationError as exc:
            st.session_state["results"] = None
            st.warning(str(exc))
        except Exception as exc:  # last-resort guard so the page never shows a traceback
            st.session_state["results"] = None
            st.error(f"Something went wrong while finding recommendations: {exc}")

    results = st.session_state.get("results")
    if results:
        prefs, found = results["prefs"], results["movies"]
        chosen = [
            f"{label}: {value}"
            for label, value in (
                ("Genre", prefs.genre),
                ("Mood", prefs.mood),
                ("Language", prefs.language),
            )
            if value != ANY
        ]
        if prefs.keywords:
            chosen.append(f"Keywords: {prefs.keywords}")

        if found.empty:
            st.markdown(
                empty_state_html(
                    "No suitable recommendations yet 🍿",
                    "Nothing in the catalogue is similar enough to "
                    f"({' · '.join(chosen)}). Try fewer filters, a different "
                    "combination, or add more movies to movies.csv.",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.subheader(f"Top {len(found)} for you")
            st.caption("Based on " + " · ".join(chosen))
            st.markdown(grid_html(found), unsafe_allow_html=True)
            st.caption(
                "The similarity score is the cosine similarity (0 to 1) between your "
                "preferences and each movie's TF-IDF profile. It shows how closely a "
                "movie's profile matches what you picked. It is not a prediction of "
                "accuracy or of how much you will enjoy the movie."
            )
    else:
        st.info("Choose your preferences above and press **Recommend Movies**.")

# ----------------------------- Search tab ---------------------------------- #
with tab_search:
    query = st.text_input(
        "Search movies by title", max_chars=100, placeholder="e.g. Inception", key="search_query"
    )
    matches = recommender.search(query)

    if query.strip() and matches.empty:
        st.markdown(
            empty_state_html(
                "No movie found",
                f"We couldn't find a title matching '{query.strip()}'. "
                "Check the spelling or browse the Catalogue tab.",
            ),
            unsafe_allow_html=True,
        )
    else:
        if query.strip():
            st.caption(f"{len(matches)} matching movie(s)")
        selected = st.selectbox("Select a movie to see its details", matches["title"].tolist())
        movie = recommender.get_movie(selected)
        if movie is not None:
            detail = with_posters(movie.to_frame().T.reset_index(drop=True)).iloc[0]
            st.markdown(detail_html(detail), unsafe_allow_html=True)

            st.subheader("Similar movies")
            with st.spinner("Finding similar movies..."):
                similar = with_posters(recommender.similar_to(movie["title"], top_n=4, min_score=0.01))
            if similar.empty:
                st.caption("No similar movies in the catalogue yet.")
            else:
                st.markdown(grid_html(similar, target="this movie's"), unsafe_allow_html=True)
                st.caption(
                    "Similar movies are found by comparing this movie's TF-IDF vector "
                    "with every other movie using cosine similarity."
                )

# ----------------------------- Catalogue tab ------------------------------- #
with tab_catalogue:
    f_genre, f_mood, f_language = st.columns(3)
    pick_genres = f_genre.multiselect("Filter by genre", recommender.genres, key="cat_genre")
    pick_moods = f_mood.multiselect("Filter by mood", recommender.moods, key="cat_mood")
    pick_languages = f_language.multiselect(
        "Filter by language", recommender.languages, key="cat_language"
    )

    def _has_any(cell: str, wanted: list) -> bool:
        if not wanted:
            return True
        have = {v.lower() for v in split_multi(cell)}
        return any(w.lower() in have for w in wanted)

    mask = (
        movies["genre"].map(lambda c: _has_any(c, pick_genres))
        & movies["mood"].map(lambda c: _has_any(c, pick_moods))
        & movies["language"].map(lambda c: _has_any(c, pick_languages))
    )
    filtered = movies[mask].reset_index(drop=True)

    st.write(f"Showing **{len(filtered)}** of **{len(movies)}** movies.")

    if filtered.empty:
        st.markdown(
            empty_state_html(
                "No movies match these filters",
                "Remove a filter to see more movies.",
            ),
            unsafe_allow_html=True,
        )
    else:
        view_mode = st.radio("View", ["Cards", "Table"], horizontal=True, key="cat_view")
        if view_mode == "Cards":
            page_size = 12
            pages = max(1, -(-len(filtered) // page_size))
            page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="cat_page")
            start = (int(page) - 1) * page_size
            with st.spinner("Loading movies..."):
                page_movies = with_posters(filtered.iloc[start : start + page_size])
            st.markdown(grid_html(page_movies, show_score=False), unsafe_allow_html=True)
            st.caption(f"Page {int(page)} of {pages}")
        else:
            table = filtered[["title", "year", "genre", "mood", "language", "description"]].rename(
                columns=str.title
            )
            st.dataframe(table, hide_index=True)

        st.download_button(
            "⬇️ Download these movies as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="movies_filtered.csv",
            mime="text/csv",
        )

    with st.expander("How to add more movies"):
        st.markdown(
            "Open `movies.csv`, add one row per movie and save. "
            "The app picks up the change automatically.\n\n"
            "- **Required columns:** `title`, `genre`, `mood`, `language`, `description`\n"
            "- **Optional columns:** `year` (4 digits), `poster_url` (a link starting with `http://` or `https://`)\n"
            "- Use `|` to give a movie several genres or moods, e.g. `Action|Drama`\n"
            "- Wrap text that contains commas in double quotes"
        )

# ----------------------------- How it works tab ---------------------------- #
with tab_how:
    st.markdown(
        """
### How the AI recommends movies

This project uses **content-based filtering**: it recommends movies whose *content*
(genre, mood, language and description) is most similar to what you ask for.
It does not need any other user's data.

1. **Movie profiles.** Each movie becomes a text profile made of its genre, mood, language
   and description. Genre is repeated three times and mood and language twice, so they count more.
2. **TF-IDF.** `TfidfVectorizer` converts every profile into a vector of numbers.
   *TF* (term frequency) rewards words that appear in a movie's profile and *IDF*
   (inverse document frequency) lowers the weight of words that appear in many movies.
3. **Your preferences** are converted into a vector with the same vectoriser.
4. **Cosine similarity** measures the angle between your vector and each movie's vector:
   `similarity = (A · B) / (|A| × |B|)`. A value near 1 means very similar and 0 means nothing in common.
5. **Ranking.** Movies are sorted by similarity and the top results are shown, together with
   the terms that contributed most to each score.
"""
    )
    st.info(
        "The similarity score shows how close a movie's profile is to your preferences. "
        "It is not a prediction accuracy and does not predict whether you will enjoy a movie."
    )

    st.markdown("#### See a movie's TF-IDF vector")
    demo_title = st.selectbox(
        "Choose a movie to inspect", movies["title"].tolist(), key="demo_movie"
    )
    st.dataframe(recommender.top_terms(demo_title, n=8), hide_index=True)
    st.caption(
        f"The model learned {catalogue_stats['vocabulary']} distinct terms from "
        f"{catalogue_stats['movies']} movies. These are the highest-weighted terms for "
        f"“{demo_title}”."
    )

if tmdb_api_key():
    st.caption(TMDB_NOTICE)
