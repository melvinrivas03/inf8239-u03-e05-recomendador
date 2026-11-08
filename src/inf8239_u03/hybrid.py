"""Recomendador híbrido con fallback documentado para usuarios fríos.

Política (en este orden):
1. Usuario conocido por la factorización y con ≥ ``min_history`` valoraciones:
       base  = alpha · minmax(MF) + (1 − alpha) · minmax(contenido)
       score = (1 − gamma) · minmax(base) + gamma · minmax(popularidad)     → fuente "híbrido"
   (gamma = 0 reproduce el híbrido de LAB09; gamma > 0 añade un prior de popularidad,
    introducido en el protocolo v2 tras observar que la popularidad dominaba el acierto)
2. Usuario sin vector colaborativo pero con algunas valoraciones (onboarding):
       score = beta · minmax(contenido) + (1 − beta) · minmax(popularidad)   → fuente "contenido+popularidad"
3. Usuario sin ninguna valoración:
       score = popularidad suavizada                                          → fuente "popularidad (NO personalizada)"
"""
from __future__ import annotations

import numpy as np

from .base import History, ItemIndex, Recommender, minmax
from .content import ContentRecommender
from .matrix_factorization import MatrixFactorization
from .popularity import PopularitySmoothed


class HybridRecommender(Recommender):
    def __init__(self, mf: MatrixFactorization, content: ContentRecommender, popularity: PopularitySmoothed,
                 alpha: float = 0.75, beta: float = 0.5, gamma: float = 0.0, min_history: int = 1):
        for nm, v in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{nm} debe estar en [0, 1]")
        self.mf, self.content, self.pop = mf, content, popularity
        self.alpha, self.beta, self.gamma, self.min_history = alpha, beta, gamma, min_history
        self.name = f"Híbrido (alpha={alpha:g}, gamma={gamma:g})"
        self.last_source = None

    def fit(self, train, index: ItemIndex):
        # los componentes se entrenan por separado; el híbrido sólo los combina
        return self

    def source_for(self, user_id, hist: History) -> str:
        if self.mf.knows_user(user_id) and len(hist) >= self.min_history:
            return "híbrido"
        if len(hist) > 0:
            return "contenido+popularidad"
        return "popularidad (NO personalizada)"

    def score(self, user_id, hist: History) -> np.ndarray:
        src = self.source_for(user_id, hist)
        self.last_source = src
        if src == "híbrido":
            cf = self.mf.score(user_id, hist)
            cf = np.where(np.isfinite(cf), cf, np.nanmin(np.where(np.isfinite(cf), cf, np.nan)))
            base = self.alpha * minmax(cf) + (1 - self.alpha) * minmax(self.content.score(user_id, hist))
            if self.gamma == 0:
                return base
            return (1 - self.gamma) * minmax(base) + self.gamma * minmax(self.pop.score(user_id, hist))
        if src == "contenido+popularidad":
            return (self.beta * minmax(self.content.score(user_id, hist))
                    + (1 - self.beta) * minmax(self.pop.score(user_id, hist)))
        return self.pop.score(user_id, hist)

    @property
    def personalized(self) -> bool:  # type: ignore[override]
        return self.last_source != "popularidad (NO personalizada)"

    def n_params(self) -> int:
        return self.mf.n_params() + self.content.n_params()
