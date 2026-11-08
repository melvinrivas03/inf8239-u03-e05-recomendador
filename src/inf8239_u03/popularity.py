"""Baselines no personalizados: aleatorio, popularidad por conteo y popularidad suavizada.

Popularidad suavizada (media bayesiana):
    weighted_score_i = (n_i / (n_i + m)) * media_i + (m / (n_i + m)) * media_global
Una película con una sola valoración alta no domina el ranking: con pocos votos
su puntuación se acerca a la media global.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import History, ItemIndex, Recommender


def popularity_table(train: pd.DataFrame, movies: pd.DataFrame, m: float | None = None) -> pd.DataFrame:
    g = train.groupby("movieId")["rating"].agg(["mean", "count"])
    C = train["rating"].mean()
    if m is None:
        m = float(g["count"].quantile(0.90))
    g["weighted_score"] = (g["count"] / (g["count"] + m)) * g["mean"] + (m / (g["count"] + m)) * C
    out = movies[["movieId", "title", "genres"]].merge(g.reset_index(), on="movieId", how="inner")
    out.attrs["m"] = m
    out.attrs["global_mean"] = C
    return out.sort_values(["weighted_score", "count"], ascending=False).reset_index(drop=True)


class RandomRecommender(Recommender):
    name = "Aleatorio"
    personalized = False

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, train, index):
        self.n = len(index)
        self.rng = np.random.default_rng(self.seed)
        return self

    def score(self, user_id, hist):
        return self.rng.random(self.n)


class PopularityCount(Recommender):
    """Ordena por número de valoraciones (lo más visto)."""
    name = "Popularidad (conteo)"
    personalized = False

    def fit(self, train, index):
        counts = train.groupby("movieId").size()
        self.counts = np.zeros(len(index))
        pos = index.to_pos(counts.index)
        self.counts[pos] = counts.to_numpy()
        return self

    def score(self, user_id, hist):
        return self.counts.copy()

    def n_params(self) -> int:
        return int(self.counts.size)


class PopularitySmoothed(Recommender):
    """Ordena por media bayesiana; también predice ratings (media suavizada del ítem)."""
    name = "Popularidad suavizada"
    personalized = False
    predicts_ratings = True

    def __init__(self, m: float | None = None):
        self.m = m

    def fit(self, train, index):
        g = train.groupby("movieId")["rating"].agg(["mean", "count"])
        self.mu = float(train["rating"].mean())
        m = float(g["count"].quantile(0.90)) if self.m is None else self.m
        self.m_ = m
        ws = (g["count"] / (g["count"] + m)) * g["mean"] + (m / (g["count"] + m)) * self.mu
        self.scores = np.full(len(index), self.mu)
        self.scores[index.to_pos(g.index)] = ws.to_numpy()
        self.counts = np.zeros(len(index))
        self.counts[index.to_pos(g.index)] = g["count"].to_numpy()
        self.index = index
        return self

    def score(self, user_id, hist):
        # en empate (p. ej. ítems sin valoraciones = media global) favorece más evidencia
        return self.scores + 1e-9 * self.counts

    def n_params(self) -> int:
        return int(self.scores.size)

    def predict(self, user_ids, movie_ids) -> np.ndarray:
        pos = np.array([self.index.pos.get(int(m), -1) for m in movie_ids])
        out = np.full(len(pos), self.mu)
        ok = pos >= 0
        out[ok] = self.scores[pos[ok]]
        return out


class GlobalMean(Recommender):
    """Sólo para RMSE: predice la media global."""
    name = "Media global"
    personalized = False
    predicts_ratings = True

    def fit(self, train, index):
        self.mu = float(train["rating"].mean())
        self.n = len(index)
        return self

    def score(self, user_id, hist):
        return np.zeros(self.n)

    def predict(self, user_ids, movie_ids):
        return np.full(len(movie_ids), self.mu)
