"""Rutas y constantes del proyecto (todas relativas a la raíz del repositorio)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
ML_DIR = DATA_RAW / "ml-latest-small"
MANIFEST = DATA_RAW / "download_manifest.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_README = "https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html"
CITATION = (
    "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. "
    "ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19. https://doi.org/10.1145/2827872"
)

SEEDS = (42, 7, 2025)
K = 10
