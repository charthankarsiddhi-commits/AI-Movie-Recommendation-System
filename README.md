# 🎬 AI Movie Recommendation System

A dark-themed, responsive web application that recommends movies by **genre, mood and language**
using **content-based filtering**: **TF-IDF** vectors and **cosine similarity** (scikit-learn).

Built with Python, Streamlit, Pandas and Scikit-learn, and ready for Streamlit Community Cloud.
College project: Computer Engineering.

## Features

- Recommendations from genre, mood and language dropdowns, plus optional description keywords
- Real **TF-IDF + cosine similarity** ranking, with a **similarity score** and a **"Why" explanation** on every card
- 34 movies in 6 languages, extendable by editing one CSV file
- Movie cards with poster, title, year, genres, mood, language and description
- Real posters when available (see [Posters](#posters)), with generated fallback artwork otherwise
- Search by title (partial matches and small typos), movie details and **similar movies**
- Catalogue with genre, mood and language filters, a movie counter, card and table views, and CSV download
- A **How it works** page that shows the actual TF-IDF weights of any movie
- Input validation, missing-data handling, loading spinners, friendly "no results" messages
- **Clear preferences** button; responsive layout for mobile and desktop

## Project structure

```
movie_recommender/
├── app.py               # Streamlit user interface (draws the page only)
├── ai_recommender.py    # The AI: dataset loading, TF-IDF, cosine similarity, validation, explanations
├── movies.csv           # Movie dataset (edit this to add movies)
├── requirements.txt     # Python dependencies
├── README.md
├── components.py        # HTML/CSS for movie cards and layout
├── posters.py           # Optional poster lookup (TMDB) used by app.py
├── tests/
│   └── test_recommender.py
├── .streamlit/config.toml   # Dark theme
└── .gitignore
```

The AI algorithm (`ai_recommender.py`) is completely separate from the UI (`app.py`).
All paths are relative to the project folder, so it works on any computer.

## Run locally

Requires Python 3.10 or newer.

```bash
# 1. Go to the project folder
cd movie_recommender

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

### Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the algorithm, the genre/mood/language filters, validation, empty results, missing or
broken CSV data, poster lookup (with a simulated network) and the HTML generation.

### Check the layout on phone and desktop

The page uses a responsive CSS grid and Streamlit's column layout. To check it:
- open the app in Chrome, press **F12**, click the **device toolbar** icon and choose a phone, or
- open the deployed link on your phone.

## Add more movies

Open `movies.csv`, add one row per movie, and save. The running app reloads the data automatically.

| Column | Required | Notes |
|---|---|---|
| `title` | yes | Must be unique |
| `genre` | yes | Use `\|` for several, e.g. `Action\|Drama` |
| `mood` | yes | e.g. `Happy`, `Romantic`, `Exciting`, `Emotional` |
| `language` | yes | e.g. `Hindi`, `English` |
| `description` | yes | One or two sentences; richer descriptions give better matches |
| `year` | no | 4 digits |
| `poster_url` | no | A link starting with `http://` or `https://` that points to an image |

Example (wrap text that contains commas in double quotes):

```csv
Chhichhore,Comedy|Drama,Happy,Hindi,2019,"A group of college friends reunite and share lessons about success and failure.",
```

New genres, moods and languages appear in the dropdowns automatically. Rows without a title are skipped,
duplicate titles are ignored and blank fields are handled gracefully.

## Posters

Poster images are chosen in this order:

1. the `poster_url` in `movies.csv`, if you fill it in;
2. a poster found through the free **TMDB** service, if you add an API key (optional);
3. generated fallback artwork (a coloured card with the title), which always works offline.

Poster images belong to their respective owners, so none are bundled in this project.
To switch on automatic posters:

1. Create a free account at <https://www.themoviedb.org> and request an API key under Settings → API.
2. **Locally:** create the file `.streamlit/secrets.toml` containing `TMDB_API_KEY = "your_key_here"`.
   (It is already listed in `.gitignore`, so it will not be uploaded to GitHub.)
3. **On Streamlit Community Cloud:** in your app's **Advanced settings → Secrets** (or the app's **Settings → Secrets**
   later), paste the same line.

If the key is missing or the internet is down, the app silently uses the fallback artwork.

## Deploy to Streamlit Community Cloud

### 1. Upload the complete project to GitHub

1. Create a free account at <https://github.com> and click **New repository**.
   Name it (for example `movie-recommender`), choose **Public**, leave "Add a README" **unchecked**, and click **Create repository**.
2. In a terminal inside the project folder:

   ```bash
   git init
   git add .
   git commit -m "AI Movie Recommendation System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/movie-recommender.git
   git push -u origin main
   ```

   GitHub does not accept your account password on the command line; sign in through the browser popup
   from Git Credential Manager, or use a [personal access token](https://github.com/settings/tokens).

   *No Git installed?* On the repository page choose **Add file → Upload files**, drag in the **contents** of the
   project folder (including the `tests` and `.streamlit` folders) and click **Commit changes**.

3. Check that `app.py`, `ai_recommender.py`, `movies.csv` and `requirements.txt` are visible in the repository root.

### 2. Connect the repository and deploy

1. Go to <https://share.streamlit.io> and sign in with GitHub (allow access to your repository when asked).
2. Click **Create app**, then **Yup, I have an app**.
3. Fill in:
   - **Repository:** `<your-username>/movie-recommender`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** optional custom name
4. Optional: open **Advanced settings** to choose a Python version (3.12 or newer is fine) and to paste secrets such as the TMDB key.
5. Deploy and wait a few minutes while the packages install. You get a public `*.streamlit.app` link to show your professor.

Every `git push` updates the live app automatically. To add movies, edit `movies.csv` on GitHub and commit.
Button labels on Streamlit's site change now and then; the official guide is at
<https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>.

## How it works (for your viva)

**Type of AI:** content-based filtering, a recommendation technique that uses *item content* rather than other users' behaviour.

1. **Dataset (Pandas).** `movies.csv` is loaded and cleaned: blank titles removed, duplicates dropped,
   missing values handled.
2. **Movie profile.** For every movie, the genre (repeated 3×), mood (2×) and language (2×) are turned into
   tokens such as `genre_science_fiction` and combined with the description text.
   Repeating a token gives it a higher term frequency, i.e. more weight.
3. **TF-IDF (`TfidfVectorizer`).** Each profile becomes a numeric vector.
   *TF* = how often a term appears in a movie; *IDF* = `log(N / df)`-style weighting that lowers the weight
   of terms that appear in many movies and raises the weight of distinctive ones.
4. **User vector.** The chosen genre, mood, language and keywords are converted to a vector with the *same* vectoriser.
5. **Cosine similarity.** `similarity = (A · B) / (‖A‖ × ‖B‖)`, the cosine of the angle between two vectors:
   1 means pointing the same way, 0 means nothing in common.
6. **Ranking.** Movies are sorted by similarity; those with almost no similarity are dropped, which is what produces the "no suitable recommendations" message.
7. **Explanation.** For each result the app lists the terms with the largest `query weight × movie weight`,
   because those terms contribute most to the cosine score.
8. **Similar movies.** The same cosine similarity is applied between one movie's vector and all the others.

### Likely viva questions

**Why content-based and not collaborative filtering?** We have no user ratings. Content-based filtering needs only movie metadata and
works for new movies immediately. The downside is that it can only recommend items similar to the input (no surprises).

**What is TF-IDF?** A weighting scheme that scores a word high if it is frequent in one document but rare across all documents.

**What is cosine similarity and why use it?** It measures the angle between vectors, so it compares *direction* (topic)
rather than length, which suits text of different lengths.

**Is the similarity score the accuracy of the model?** No. It only says how close a movie's profile is to your preferences.
Measuring recommendation accuracy would need real user feedback (e.g. precision@k on rated data).

**Why can a perfect match score below 1?** The movie's description adds words that your query does not contain,
so the vectors do not point in exactly the same direction. A movie with a shorter description can also score slightly higher than an equally good match with a longer one.

**How do I add a movie?** Add a row to `movies.csv`; the model is rebuilt automatically.

**Limitations and future work:** small dataset; it ignores user history, ratings and popularity; keyword matching is lexical
(no understanding of synonyms). Improvements: word embeddings or sentence-transformers, user ratings (hybrid recommender), a bigger dataset.

**Is the language of RRR Hindi?** RRR was made in Telugu and released in Hindi too; the dataset follows the project brief
and lists it as Hindi. Edit the `language` column if you prefer Telugu.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment or run `python -m streamlit run app.py` |
| `ModuleNotFoundError` on Cloud | Make sure `requirements.txt` is in the repository root and committed |
| "Could not load the movie dataset" | Check that `movies.csv` is next to `app.py` and has the columns `title, genre, mood, language, description` |
| Posters do not show | Without a TMDB key the app uses fallback artwork. A `poster_url` must start with `http://` or `https://` and point directly to an image |
| Changes to the CSV do not appear | Refresh the page; the app detects the new file automatically |

## Tech stack

Python · Streamlit · Pandas · Scikit-learn (`TfidfVectorizer`, `cosine_similarity`) · NumPy · optional TMDB API for posters

*This product uses the TMDB API but is not endorsed or certified by TMDB.* (applies only when a TMDB key is configured)
