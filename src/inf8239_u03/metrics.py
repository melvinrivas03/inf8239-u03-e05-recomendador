"""Métricas de ratings, ranking y exposición del catálogo."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def hit_at_k(recs: np.ndarray, relevant: set | int, k: int) -> float:
    rel = {relevant} if np.isscalar(relevant) else set(relevant)
    return float(any(int(x) in rel for x in recs[:k]))


def precision_at_k(recs: np.ndarray, relevant: set | int, k: int) -> float:
    rel = {relevant} if np.isscalar(relevant) else set(relevant)
    return sum(int(x) in rel for x in recs[:k]) / k


def recall_at_k(recs: np.ndarray, relevant: set | int, k: int) -> float:
    rel = {relevant} if np.isscalar(relevant) else set(relevant)
    return sum(int(x) in rel for x in recs[:k]) / max(1, len(rel))


def ndcg_at_k(recs: np.ndarray, relevant: set | int, k: int) -> float:
    rel = {relevant} if np.isscalar(relevant) else set(relevant)
    dcg = sum(1 / np.log2(r + 2) for r, x in enumerate(recs[:k]) if int(x) in rel)
    idcg = sum(1 / np.log2(r + 2) for r in range(min(k, len(rel))))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(recs: np.ndarray, relevant: set | int, k: int) -> float:
    rel = {relevant} if np.isscalar(relevant) else set(relevant)
    for r, x in enumerate(recs[:k]):
        if int(x) in rel:
            return 1.0 / (r + 1)
    return 0.0


def catalog_coverage(rec_lists, n_items: int) -> float:
    """Proporción del catálogo que aparece al menos una vez en alguna lista."""
    seen = set()
    for recs in rec_lists:
        seen.update(int(x) for x in recs)
    return len(seen) / n_items if n_items else 0.0


def novelty(rec_lists, item_pop: np.ndarray, n_users: int) -> float:
    """Media de −log2(p_i), p_i = fracción de usuarios que valoró el ítem en entrenamiento."""
    p = np.clip(item_pop / max(1, n_users), 1 / max(1, n_users), 1.0)
    vals = [-np.log2(p[int(x)]) for recs in rec_lists for x in recs]
    return float(np.mean(vals)) if vals else 0.0


def head_share(rec_lists, head_items: set) -> float:
    """Fracción de posiciones recomendadas ocupadas por ítems de la «cabeza» (más populares)."""
    flat = [int(x) for recs in rec_lists for x in recs]
    return float(np.mean([x in head_items for x in flat])) if flat else 0.0


def intra_list_diversity(rec_lists, X: np.ndarray) -> float:
    """1 − similitud coseno media entre pares de ítems dentro de cada lista (X con filas L2)."""
    vals = []
    for recs in rec_lists:
        recs = np.asarray(recs, dtype=int)
        if len(recs) < 2:
            continue
        S = X[recs] @ X[recs].T
        n = len(recs)
        vals.append(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))
    return float(np.mean(vals)) if vals else 0.0


def gini(counts: np.ndarray) -> float:
    """Gini de la exposición (0 = uniforme, 1 = todo en un ítem)."""
    x = np.sort(np.asarray(counts, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def pareto_mask(df, maximize: list[str], minimize: list[str]) -> np.ndarray:
    """True si la fila NO está dominada.

    Una configuración está dominada si otra es igual o mejor en todo (utilidad ≥, costo ≤)
    y estrictamente mejor en al menos un criterio.
    """
    vals_max = df[maximize].to_numpy(float) if maximize else np.empty((len(df), 0))
    vals_min = df[minimize].to_numpy(float) if minimize else np.empty((len(df), 0))
    keep = np.ones(len(df), dtype=bool)
    for a in range(len(df)):
        for b in range(len(df)):
            if a == b:
                continue
            ge = np.all(vals_max[b] >= vals_max[a]) and np.all(vals_min[b] <= vals_min[a])
            gt = np.any(vals_max[b] > vals_max[a]) or np.any(vals_min[b] < vals_min[a])
            if ge and gt:
                keep[a] = False
                break
    return keep
