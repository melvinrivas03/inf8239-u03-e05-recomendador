"""Descarga MovieLens Latest Small desde GroupLens (o instala un zip local) y registra su hash.

Uso:
    python scripts/download_data.py                       # descarga oficial (~1 MB)
    python scripts/download_data.py --zip ~/Descargas/ml-latest-small.zip
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inf8239_u03 import config  # noqa: E402
from inf8239_u03.data import download_movielens, install_zip  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--zip", type=Path, help="ruta a ml-latest-small.zip ya descargado de GroupLens")
a = p.parse_args()
if a.zip:
    man = install_zip(a.zip, source=f"zip local obtenido de {config.MOVIELENS_URL}")
else:
    man = download_movielens()
print(json.dumps(man, indent=2, ensure_ascii=False))
print("\nUso sujeto al README oficial de GroupLens:", config.MOVIELENS_README)
print("No redistribuya el dataset; data/raw/ está en .gitignore.")
