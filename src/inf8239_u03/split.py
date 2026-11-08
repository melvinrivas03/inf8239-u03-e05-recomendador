"""Particiones temporales.

* ``leave_last_out``: por usuario, la última valoración (por timestamp) va a prueba,
  la penúltima a validación y el resto a entrenamiento. Así el modelo nunca usa
  el futuro de un usuario para predecir su pasado.
* ``cold_user_split``: separa un subconjunto de usuarios que se tratan como NUEVOS:
  todas sus interacciones se eliminan del entrenamiento; se conservan sus primeras
  ``n_onboarding`` valoraciones (formulario de bienvenida) y se evalúa la siguiente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def _order(r: pd.DataFrame) -> pd.DataFrame:
    # desempate determinista por movieId cuando dos valoraciones comparten timestamp
    return r.sort_values(["userId", "timestamp", "movieId"], kind="mergesort")


def leave_last_out(ratings: pd.DataFrame, min_train: int = 3) -> Split:
    r = _order(ratings)
    pos_from_end = r.groupby("userId").cumcount(ascending=False)
    n_user = r.groupby("userId")["movieId"].transform("size")
    eligible = n_user >= min_train + 2
    test = r[eligible & (pos_from_end == 0)]
    valid = r[eligible & (pos_from_end == 1)]
    train = r[~(eligible & (pos_from_end <= 1))]
    return Split(train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True))


def merge_train_valid(s: Split) -> pd.DataFrame:
    """Entrenamiento final (train + validación) para la evaluación única en prueba."""
    return pd.concat([s.train, s.valid], ignore_index=True)


@dataclass
class ColdSplit:
    train: pd.DataFrame          # interacciones de usuarios "cálidos" (sin el ítem de prueba)
    onboarding: pd.DataFrame     # primeras n valoraciones de los usuarios fríos
    test: pd.DataFrame           # última valoración de cada usuario frío
    cold_users: np.ndarray


def cold_user_split(ratings: pd.DataFrame, frac: float = 0.10, n_onboarding: int = 3,
                    seed: int = 42, exclude=None) -> ColdSplit:
    """``exclude``: usuarios que no pueden ser elegidos como fríos (p. ej. para una validación disjunta)."""
    r = _order(ratings)
    all_users = np.sort(r.userId.unique())
    users = all_users if exclude is None else np.setdiff1d(all_users, np.asarray(list(exclude)))
    frac = frac * len(all_users) / max(1, len(users))  # mismo número absoluto de usuarios fríos
    rng = np.random.default_rng(seed)
    cold = np.sort(rng.choice(users, size=max(1, int(round(frac * len(users)))), replace=False))
    is_cold = r.userId.isin(cold)
    cold_r = r[is_cold]
    pos = cold_r.groupby("userId").cumcount()
    # objetivo = la interacción INMEDIATAMENTE posterior al onboarding (lo que haría un usuario nuevo),
    # no su última valoración años después.
    return ColdSplit(
        train=r[~is_cold].reset_index(drop=True),
        onboarding=cold_r[pos < n_onboarding].reset_index(drop=True),
        test=cold_r[pos == n_onboarding].reset_index(drop=True),
        cold_users=cold,
    )
