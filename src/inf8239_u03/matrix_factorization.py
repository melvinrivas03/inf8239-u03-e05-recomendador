"""Factorización matricial con sesgos (Funk-SVD) entrenada por SGD.

    r̂_ui = μ + b_u + b_i + p_u · q_i

Minimiza el error cuadrático en ratings observados con regularización L2.
Usuarios o ítems que no aparecen en entrenamiento no tienen vector aprendido:
para un usuario nuevo la señal colaborativa NO existe (``knows_user`` = False)
y el híbrido debe usar un fallback.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .base import History, ItemIndex, Recommender


class BiasBaseline(Recommender):
    """μ + b_u + b_i con sesgos regularizados (mínimos cuadrados alternos, pocas iteraciones)."""
    name = "Sesgos usuario+ítem"
    predicts_ratings = True

    def __init__(self, reg_u: float = 10.0, reg_i: float = 15.0, iters: int = 5):
        self.reg_u, self.reg_i, self.iters = reg_u, reg_i, iters

    def fit(self, train, index):
        self.index = index
        self.mu = float(train.rating.mean())
        users = np.sort(train.userId.unique())
        self.upos = {int(u): j for j, u in enumerate(users)}
        u = train.userId.map(self.upos).to_numpy()
        i = index.to_pos(train.movieId)
        r = train.rating.to_numpy() - self.mu
        self.bu, self.bi = np.zeros(len(users)), np.zeros(len(index))
        for _ in range(self.iters):
            self.bi = np.bincount(i, r - self.bu[u], len(index)) / (np.bincount(i, minlength=len(index)) + self.reg_i)
            self.bu = np.bincount(u, r - self.bi[i], len(users)) / (np.bincount(u, minlength=len(users)) + self.reg_u)
        return self

    def score(self, user_id, hist):
        return self.mu + self.bi

    def predict(self, user_ids, movie_ids):
        bu = np.array([self.bu[self.upos[int(x)]] if int(x) in self.upos else 0.0 for x in user_ids])
        bi = np.array([self.bi[self.index.pos[int(m)]] if int(m) in self.index.pos else 0.0 for m in movie_ids])
        return np.clip(self.mu + bu + bi, 0.5, 5.0)


class MatrixFactorization(Recommender):
    name = "Factorización (MF-SGD)"
    predicts_ratings = True

    def __init__(self, factors: int = 20, epochs: int = 12, lr: float = 0.01, reg: float = 0.05,
                 init_std: float = 0.1, seed: int = 42):
        self.factors, self.epochs, self.lr, self.reg = factors, epochs, lr, reg
        self.init_std, self.seed = init_std, seed

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, train: pd.DataFrame, index: ItemIndex):
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        self.index = index
        users = np.sort(train.userId.unique())
        self.upos = {int(u): j for j, u in enumerate(users)}
        u = train.userId.map(self.upos).to_numpy()
        i = index.to_pos(train.movieId)
        r = train.rating.to_numpy(dtype=float)
        self.mu = float(r.mean())
        nU, nI, f = len(users), len(index), self.factors
        P = rng.normal(0, self.init_std, (nU, f))
        Q = rng.normal(0, self.init_std, (nI, f))
        bu, bi = np.zeros(nU), np.zeros(nI)
        self.item_seen = np.zeros(nI, dtype=bool)
        self.item_seen[i] = True
        lr, reg = self.lr, self.reg
        self.history_ = []
        for ep in range(self.epochs):
            order = rng.permutation(len(r))
            sq = 0.0
            for idx in order:
                uu, ii = u[idx], i[idx]
                pu, qi = P[uu], Q[ii]
                e = r[idx] - (self.mu + bu[uu] + bi[ii] + pu @ qi)
                sq += e * e
                bu[uu] += lr * (e - reg * bu[uu])
                bi[ii] += lr * (e - reg * bi[ii])
                P[uu] = pu + lr * (e * qi - reg * pu)
                Q[ii] = qi + lr * (e * pu - reg * qi)
            self.history_.append(float(np.sqrt(sq / len(r))))
        self.P, self.Q, self.bu, self.bi = P, Q, bu, bi
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ inferencia
    def knows_user(self, user_id) -> bool:
        return user_id is not None and int(user_id) in self.upos

    def score(self, user_id, hist: History):
        if not self.knows_user(user_id):
            return np.full(len(self.index), np.nan)
        j = self.upos[int(user_id)]
        s = self.mu + self.bu[j] + self.bi + self.Q @ self.P[j]
        # ítems nunca vistos en entrenamiento: su vector es ruido de inicialización
        s[~self.item_seen] = -np.inf
        return s

    def predict(self, user_ids, movie_ids):
        out = np.empty(len(user_ids))
        for n, (uid, mid) in enumerate(zip(user_ids, movie_ids)):
            j = self.upos.get(int(uid), -1)
            k = self.index.pos.get(int(mid), -1)
            v = self.mu
            if j >= 0:
                v += self.bu[j]
            if k >= 0 and self.item_seen[k]:
                v += self.bi[k]
                if j >= 0:
                    v += self.P[j] @ self.Q[k]
            out[n] = v
        return np.clip(out, 0.5, 5.0)

    def n_params(self) -> int:
        return int(self.P.size + self.Q.size + self.bu.size + self.bi.size + 1)

    def size_bytes(self) -> int:
        return int(self.P.nbytes + self.Q.nbytes + self.bu.nbytes + self.bi.nbytes + 8)
