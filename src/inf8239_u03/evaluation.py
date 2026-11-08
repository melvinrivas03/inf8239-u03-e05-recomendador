"""Evaluación offline: ratings (RMSE/MAE) y ranking Top-K sobre TODO el catálogo."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import metrics as M
from .base import History, ItemIndex, Recommender, histories


def item_counts(train: pd.DataFrame, index: ItemIndex) -> np.ndarray:
    c = train.groupby("movieId").size()
    out = np.zeros(len(index))
    out[index.to_pos(c.index)] = c.to_numpy()
    return out


def evaluate_ratings(model: Recommender, test: pd.DataFrame) -> dict:
    if not model.predicts_ratings:
        return {"RMSE": np.nan, "MAE": np.nan}
    pred = model.predict(test.userId.to_numpy(), test.movieId.to_numpy())
    return {"RMSE": M.rmse(test.rating, pred), "MAE": M.mae(test.rating, pred)}


def evaluate_ranking(model: Recommender, train: pd.DataFrame, test: pd.DataFrame, index: ItemIndex,
                     X_content: np.ndarray, k: int = 10, extra_history: pd.DataFrame | None = None,
                     return_lists: bool = False) -> dict:
    """Leave-last-out: para cada usuario de ``test`` se ordena todo el catálogo excepto lo que ya vio.

    ``extra_history`` permite pasar el historial de onboarding de usuarios fríos (que no está en train).
    """
    hist = histories(pd.concat([train, extra_history]) if extra_history is not None else train, index)
    counts = item_counts(train, index)
    n_train_users = train.userId.nunique()
    head = set(np.argsort(-counts)[: max(1, int(0.10 * (counts > 0).sum()))].tolist())
    tb = counts  # desempate por popularidad, documentado
    rows, lists, sources = [], [], []
    t0 = time.perf_counter()
    for uid, mid in zip(test.userId.to_numpy(), test.movieId.to_numpy()):
        h = hist.get(int(uid), History())
        recs = model.recommend(int(uid), h, k=k, tiebreak=tb)
        target = index.pos[int(mid)]
        lists.append(recs)
        sources.append(getattr(model, "last_source", None))
        rows.append((M.hit_at_k(recs, target, k), M.precision_at_k(recs, target, k),
                     M.ndcg_at_k(recs, target, k), M.mrr_at_k(recs, target, k)))
    secs = time.perf_counter() - t0
    arr = np.array(rows) if rows else np.zeros((0, 4))
    exposure = np.bincount(np.concatenate(lists).astype(int), minlength=len(index)) if lists else np.zeros(len(index))
    out = {
        f"HitRate@{k}": float(arr[:, 0].mean()),
        f"HitRate@{k}_EE": float(np.sqrt(arr[:, 0].mean() * (1 - arr[:, 0].mean()) / max(1, len(arr)))),
        f"Precision@{k}": float(arr[:, 1].mean()),
        f"NDCG@{k}": float(arr[:, 2].mean()),
        f"MRR@{k}": float(arr[:, 3].mean()),
        "Cobertura": M.catalog_coverage(lists, len(index)),
        "Novedad": M.novelty(lists, counts, n_train_users),
        "Cuota_cabeza": M.head_share(lists, head),
        "Diversidad_ILD": M.intra_list_diversity(lists, X_content),
        "Gini_exposicion": M.gini(exposure),
        "usuarios_eval": len(rows),
        "seg_inferencia": secs,
    }
    if return_lists:
        out["_lists"] = lists
        out["_sources"] = sources
    return out


def hits_by_user(model, train, test, index, k=10) -> pd.Series:
    """HitRate@K por usuario (para analizar subgrupos por tamaño de historial)."""
    hist = histories(train, index)
    counts = item_counts(train, index)
    out = {}
    for uid, mid in zip(test.userId.to_numpy(), test.movieId.to_numpy()):
        recs = model.recommend(int(uid), hist.get(int(uid), History()), k=k, tiebreak=counts)
        out[int(uid)] = M.hit_at_k(recs, index.pos[int(mid)], k)
    return pd.Series(out, name="hit")


def paired_comparison(hits_a: pd.Series, hits_b: pd.Series, n_boot: int = 5000, seed: int = 0) -> dict:
    """Comparación pareada por usuario de dos modelos (mismos usuarios de prueba).

    Devuelve la diferencia de HitRate (a − b), su IC 95 % por bootstrap y la prueba exacta de
    McNemar sobre los pares discordantes.
    """
    from scipy.stats import binomtest

    a, b = hits_a.align(hits_b, join="inner")
    d = (a - b).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(axis=1)
    a_only, b_only = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"dif_HitRate": float(d.mean()), "IC95_inf": float(np.percentile(boots, 2.5)),
            "IC95_sup": float(np.percentile(boots, 97.5)), "aciertos_solo_A": a_only,
            "aciertos_solo_B": b_only, "p_McNemar": float(p), "usuarios": len(d)}
