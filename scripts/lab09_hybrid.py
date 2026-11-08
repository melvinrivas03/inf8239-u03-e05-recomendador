"""Experimento individual de LAB09: una configuración MF + híbrido, evaluada en validación.

    python scripts/lab09_hybrid.py --factors 20 --epochs 12 --alpha 0.75 --seed 42
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inf8239_u03.base import ItemIndex, histories  # noqa: E402
from inf8239_u03.content import ContentRecommender  # noqa: E402
from inf8239_u03.data import load_movielens  # noqa: E402
from inf8239_u03.evaluation import evaluate_ranking, evaluate_ratings, item_counts  # noqa: E402
from inf8239_u03.hybrid import HybridRecommender  # noqa: E402
from inf8239_u03.matrix_factorization import MatrixFactorization  # noqa: E402
from inf8239_u03.popularity import PopularityCount  # noqa: E402
from inf8239_u03.split import leave_last_out  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--factors", type=int, default=20)
p.add_argument("--epochs", type=int, default=12)
p.add_argument("--alpha", type=float, default=0.75)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

ml = load_movielens()
idx = ItemIndex(ml.movies)
s = leave_last_out(ml.ratings)
content = ContentRecommender().fit(None, idx, ml.movies)
pop = PopularityCount().fit(s.train, idx)
mf = MatrixFactorization(a.factors, a.epochs, seed=a.seed).fit(s.train, idx)
hyb = HybridRecommender(mf, content, pop, alpha=a.alpha)
r_mf = evaluate_ranking(mf, s.train, s.valid, idx, content.X)
r_h = evaluate_ranking(hyb, s.train, s.valid, idx, content.X)
print(f"MF f={a.factors} ep={a.epochs} seed={a.seed}: RMSE_val={evaluate_ratings(mf, s.valid)['RMSE']:.4f} "
      f"HitRate@10={r_mf['HitRate@10']:.4f} cobertura={r_mf['Cobertura']:.4f} "
      f"tiempo={mf.fit_seconds_:.1f}s tamaño≈{mf.size_bytes()/1024:.0f} KB")
print(f"Híbrido alpha={a.alpha}: HitRate@10={r_h['HitRate@10']:.4f} cobertura={r_h['Cobertura']:.4f}")
u = int(s.train.userId.value_counts().idxmax())
h = histories(s.train, idx)[u]
print(f"\nEjemplo usuario {u} ({len(h)} valoraciones):")
for r, p_ in enumerate(hyb.recommend(u, h, 10, tiebreak=item_counts(s.train, idx)), 1):
    print(f"  {r:>2}. {idx.titles[p_]}  [{idx.genres[p_]}]")
