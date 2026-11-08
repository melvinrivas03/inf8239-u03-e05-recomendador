"""Recomendador por contenido: TF-IDF de géneros + similitud coseno.

La similitud significa «comparte características registradas» (géneros), no «le gustará».
El perfil de usuario es la suma de vectores de ítems vistos ponderada por
(rating - media del usuario): lo valorado por encima de su media empuja hacia
esos géneros y lo valorado por debajo aleja.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import History, ItemIndex, Recommender, top_k


def genre_matrix(movies: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    docs = movies["genres"].fillna("").str.replace("(no genres listed)", "", regex=False)
    vec = TfidfVectorizer(tokenizer=lambda s: [t for t in s.split("|") if t],
                          token_pattern=None, lowercase=False)
    X = vec.fit_transform(docs).toarray()
    return normalize(X), list(vec.get_feature_names_out())


class ContentRecommender(Recommender):
    name = "Contenido (TF-IDF géneros)"

    def fit(self, train, index: ItemIndex, movies: pd.DataFrame | None = None):
        if movies is None:
            movies = pd.DataFrame({"movieId": index.movie_ids, "title": index.titles, "genres": index.genres})
        self.X, self.vocab = genre_matrix(movies)
        self.index = index
        return self

    def profile(self, hist: History) -> np.ndarray:
        if len(hist) == 0:
            return np.zeros(self.X.shape[1])
        w = hist.ratings - hist.ratings.mean()
        if np.allclose(w, 0):
            w = np.ones_like(w)
        return w @ self.X[hist.items]

    def score(self, user_id, hist):
        p = self.profile(hist)
        n = np.linalg.norm(p)
        return self.X @ (p / n) if n > 0 else np.zeros(len(self.index))

    def similar_items(self, movie_id: int, k: int = 10, tiebreak: np.ndarray | None = None) -> pd.DataFrame:
        i = self.index.pos[int(movie_id)]
        sims = self.X @ self.X[i]
        idx = top_k(sims, k, exclude=np.array([i]), tiebreak=tiebreak)
        return pd.DataFrame({
            "query_movieId": movie_id,
            "query_title": self.index.titles[i],
            "rank": np.arange(1, len(idx) + 1),
            "movieId": self.index.movie_ids[idx],
            "title": self.index.titles[idx],
            "genres": self.index.genres[idx],
            "similarity": np.round(sims[idx], 4),
        })

    def n_params(self) -> int:
        return int(self.X.size)
