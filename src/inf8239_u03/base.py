"""Utilidades comunes: índice de ítems, historial de usuario y selección Top-K."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ItemIndex:
    """Mapea movieId <-> posición. El universo de candidatos es TODO el catálogo de movies.csv."""

    def __init__(self, movies: pd.DataFrame):
        self.movie_ids = movies["movieId"].to_numpy()
        self.pos = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.titles = movies["title"].to_numpy()
        self.genres = movies["genres"].to_numpy()

    def __len__(self) -> int:
        return len(self.movie_ids)

    def to_pos(self, movie_ids) -> np.ndarray:
        return np.fromiter((self.pos[int(m)] for m in movie_ids), dtype=np.int64)


@dataclass
class History:
    """Historial observado de un usuario (posiciones de ítem y ratings)."""
    items: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    ratings: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))

    def __len__(self) -> int:
        return len(self.items)


def histories(df: pd.DataFrame, index: ItemIndex) -> dict[int, History]:
    out: dict[int, History] = {}
    for uid, g in df.groupby("userId", sort=True):
        out[int(uid)] = History(index.to_pos(g.movieId), g.rating.to_numpy(dtype=float))
    return out


def top_k(scores: np.ndarray, k: int, exclude: np.ndarray | None = None,
          tiebreak: np.ndarray | None = None) -> np.ndarray:
    """Posiciones de los k mejores puntajes, excluyendo ítems vistos.

    Empates: se rompen por ``tiebreak`` (mayor primero, p. ej. popularidad) y luego por
    posición, para que el resultado sea determinista y explicable.
    """
    s = scores.astype(float).copy()
    if exclude is not None and len(exclude):
        s[exclude] = -np.inf
    tb = np.zeros_like(s) if tiebreak is None else tiebreak
    order = np.lexsort((np.arange(len(s)), -tb, -s))
    order = order[np.isfinite(s[order])]
    return order[:k]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


class Recommender:
    """Interfaz mínima: ``fit`` y ``score`` sobre todo el catálogo."""
    name = "base"
    personalized = True
    predicts_ratings = False

    def fit(self, train: pd.DataFrame, index: ItemIndex) -> "Recommender":  # pragma: no cover
        raise NotImplementedError

    def score(self, user_id: int | None, hist: History) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def recommend(self, user_id: int | None, hist: History, k: int = 10,
                  tiebreak: np.ndarray | None = None) -> np.ndarray:
        return top_k(self.score(user_id, hist), k, exclude=hist.items, tiebreak=tiebreak)

    def n_params(self) -> int:
        return 0
