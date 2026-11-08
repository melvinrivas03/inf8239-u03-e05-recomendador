"""Obtención, carga y auditoría de MovieLens Latest Small.

El dataset NO se redistribuye en el repositorio: se descarga desde GroupLens
(o se toma de un zip local obtenido por la persona usuaria) y se registra la
fecha, la URL y el hash SHA-256 de los bytes en ``data/raw/download_manifest.json``.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RATING_COLS = ["userId", "movieId", "rating", "timestamp"]
MOVIE_COLS = ["movieId", "title", "genres"]


@dataclass
class MovieLens:
    ratings: pd.DataFrame
    movies: pd.DataFrame


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_zip(zip_path: Path, source: str, dest: Path = config.DATA_RAW) -> dict:
    """Extrae ``ml-latest-small.zip`` en ``dest`` y escribe el manifiesto de procedencia."""
    zip_path = Path(zip_path)
    dest.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if not any(n.endswith("ratings.csv") for n in names):
            raise ValueError("El zip no contiene ratings.csv; ¿es ml-latest-small.zip?")
        zf.extractall(dest)
    manifest = {
        "dataset": "MovieLens Latest Small (ml-latest-small)",
        "source": source,
        "official_url": config.MOVIELENS_URL,
        "license_readme": config.MOVIELENS_README,
        "sha256_zip": digest,
        "bytes": zip_path.stat().st_size,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {n: None for n in names if n.endswith(".csv")},
    }
    for n in list(manifest["files"]):
        manifest["files"][n] = sha256_file(dest / n)
    (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_movielens(dest: Path = config.DATA_RAW, url: str = config.MOVIELENS_URL) -> dict:
    """Descarga el zip oficial de GroupLens y lo instala (≈1 MB)."""
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "ml-latest-small.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "inf8239-u03-e05/1.0 (academic)"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    return install_zip(tmp, source=url, dest=dest)


def load_movielens(directory: Path = config.ML_DIR) -> MovieLens:
    directory = Path(directory)
    if not (directory / "ratings.csv").exists():
        raise FileNotFoundError(
            f"No se encontró {directory/'ratings.csv'}. Ejecute `python scripts/download_data.py` "
            "o `python scripts/download_data.py --zip ruta/ml-latest-small.zip`."
        )
    ratings = pd.read_csv(directory / "ratings.csv")
    movies = pd.read_csv(directory / "movies.csv")
    return MovieLens(ratings=ratings, movies=movies)


def audit(ml: MovieLens) -> dict:
    """Comprobaciones de contrato + estadísticos descriptivos. Lanza AssertionError si algo falla."""
    r, m = ml.ratings, ml.movies
    assert list(r.columns[:4]) == RATING_COLS, f"columnas de ratings inesperadas: {list(r.columns)}"
    assert all(c in m.columns for c in MOVIE_COLS), f"columnas de movies inesperadas: {list(m.columns)}"
    assert r[RATING_COLS].notna().all().all(), "hay nulos en ratings"
    assert m["movieId"].is_unique, "movieId duplicado en movies"
    assert not r.duplicated(["userId", "movieId"]).any(), "par usuario-película duplicado"
    fk_ok = r["movieId"].isin(m["movieId"]).all()
    assert fk_ok, "ratings referencia movieId inexistente"
    assert r["rating"].between(0.5, 5.0).all(), "rating fuera de [0.5, 5]"
    assert ((r["rating"] * 2) % 1 == 0).all(), "rating no múltiplo de 0.5"
    assert (r["timestamp"] > 0).all(), "timestamp no positivo"

    n_u, n_i, n = r.userId.nunique(), r.movieId.nunique(), len(r)
    per_user = r.groupby("userId").size()
    per_item = r.groupby("movieId").size()
    ts = pd.to_datetime(r["timestamp"], unit="s", utc=True)
    return {
        "usuarios": int(n_u),
        "peliculas_catalogo": int(m.movieId.nunique()),
        "peliculas_valoradas": int(n_i),
        "valoraciones": int(n),
        "densidad": n / (n_u * n_i),
        "rating_medio": float(r.rating.mean()),
        "min_valoraciones_por_usuario": int(per_user.min()),
        "mediana_valoraciones_por_usuario": float(per_user.median()),
        "peliculas_con_1_valoracion": int((per_item == 1).sum()),
        "pct_valoraciones_top10pct_items": float(
            per_item.sort_values(ascending=False).head(max(1, int(0.1 * n_i))).sum() / n
        ),
        "fecha_min": ts.min().date().isoformat(),
        "fecha_max": ts.max().date().isoformat(),
        "peliculas_sin_genero": int((m.genres == "(no genres listed)").sum()),
        "claves_foraneas_ok": bool(fk_ok),
    }


def make_synthetic_movielens(n_users: int = 120, n_items: int = 200, seed: int = 0,
                             min_per_user: int = 20, max_per_user: int = 60) -> MovieLens:
    """Datos sintéticos con el MISMO esquema que MovieLens, sólo para pruebas.

    Los gustos dependen de géneros latentes y la popularidad sigue una ley de potencia,
    de modo que los modelos tienen señal real que aprender.
    """
    rng = np.random.default_rng(seed)
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]
    item_g = rng.random((n_items, len(genres))) < 0.25
    item_g[np.arange(n_items), rng.integers(0, len(genres), n_items)] = True
    user_pref = rng.normal(0, 1, (n_users, len(genres)))
    pop = 1 / np.arange(1, n_items + 1) ** 0.8
    rng.shuffle(pop)
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_items + 1) * 3,  # ids no contiguos, como en MovieLens
        "title": [f"Película {i} ({1980 + i % 40})" for i in range(n_items)],
        "genres": ["|".join(g for g, on in zip(genres, row) if on) for row in item_g],
    })
    rows = []
    t0 = 1_500_000_000
    for u in range(n_users):
        affinity = item_g @ user_pref[u]
        w = pop * np.exp(0.6 * affinity)
        w /= w.sum()
        k = int(rng.integers(min_per_user, max_per_user + 1))
        items = rng.choice(n_items, size=k, replace=False, p=w)
        base = 3.5 + 0.5 * affinity[items] / (item_g[items].sum(1) ** 0.5)
        rating = np.clip(np.round((base + rng.normal(0, 0.6, k)) * 2) / 2, 0.5, 5.0)
        times = t0 + np.sort(rng.integers(0, 50_000_000, k))
        for it, ra, ti in zip(items, rating, times):
            rows.append((u + 1, int(movies.movieId[it]), float(ra), int(ti)))
    ratings = pd.DataFrame(rows, columns=RATING_COLS)
    return MovieLens(ratings=ratings, movies=movies)
