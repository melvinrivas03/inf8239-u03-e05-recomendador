"""Ejecuta el experimento completo del Ejercicio 05 y escribe todo en reports/.

    python scripts/run_experiments.py            # protocolo completo (~5-10 min en CPU)
    python scripts/run_experiments.py --quick    # rejilla reducida para verificar
    python scripts/run_experiments.py --synthetic --quick   # sin MovieLens (sólo arquitectura)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inf8239_u03 import config  # noqa: E402
from inf8239_u03.data import load_movielens, make_synthetic_movielens  # noqa: E402
from inf8239_u03.pipeline import run_all  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--synthetic", action="store_true", help="datos sintéticos (no MovieLens)")
p.add_argument("--out", type=Path, default=None)
a = p.parse_args()
ml = make_synthetic_movielens(150, 300, seed=0) if a.synthetic else load_movielens()
out = a.out or (config.ROOT / "reports" / "synthetic" if a.synthetic else config.REPORTS)
run_all(ml, out=out, quick=a.quick)
