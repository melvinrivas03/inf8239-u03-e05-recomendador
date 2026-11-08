import json

import pandas as pd

from inf8239_u03.data import make_synthetic_movielens
from inf8239_u03.pipeline import run_all


def test_pipeline_end_to_end_quick(tmp_path):
    ml = make_synthetic_movielens(60, 120, seed=5)
    res = run_all(ml, out=tmp_path, quick=True, seeds=(1, 2, 3), log=lambda *a: None)
    for f in ["audit.json", "popularity_top10.csv", "content_recommendations.csv", "validation_pareto.csv",
              "test_results.csv", "cold_start.csv", "profiles.csv", "summary.json",
              "figures/pareto_validacion.png"]:
        assert (tmp_path / f).exists(), f
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["split"]["sin_fuga_temporal_por_usuario"]
    prof = pd.read_csv(tmp_path / "profiles.csv")
    assert set(prof.perfil) == {"Historial amplio", "Historial pequeño", "Usuario nuevo"}
    assert not prof.visto_antes.any()
    new = prof[prof.perfil == "Usuario nuevo"]
    assert (~new.personalizada).all() and len(new) == 10
    assert res["val"].pareto.any()
