import numpy as np
import pandas as pd
import pytest

from inf8239_u03 import metrics as M


def test_rmse_mae():
    assert M.rmse([1, 2, 3], [1, 2, 3]) == 0
    assert np.isclose(M.rmse([0, 0], [3, 4]), np.sqrt(12.5))
    assert np.isclose(M.mae([0, 0], [3, -4]), 3.5)


@pytest.mark.parametrize("recs,target,hit,ndcg,mrr", [
    ([5, 3, 9], 5, 1.0, 1.0, 1.0),
    ([5, 3, 9], 9, 1.0, 1 / np.log2(4), 1 / 3),
    ([5, 3, 9], 7, 0.0, 0.0, 0.0),
])
def test_ranking_metrics_single_target(recs, target, hit, ndcg, mrr):
    r = np.array(recs)
    assert M.hit_at_k(r, target, 3) == hit
    assert np.isclose(M.ndcg_at_k(r, target, 3), ndcg)
    assert np.isclose(M.mrr_at_k(r, target, 3), mrr)
    assert np.isclose(M.precision_at_k(r, target, 3), hit / 3)


def test_hit_respects_k():
    assert M.hit_at_k(np.array([1, 2, 3]), 3, k=2) == 0.0


def test_coverage_and_head_share():
    lists = [np.array([0, 1]), np.array([1, 2])]
    assert M.catalog_coverage(lists, 10) == 0.3
    assert M.head_share(lists, {1}) == 0.5


def test_gini_bounds():
    assert np.isclose(M.gini(np.ones(10)), 0.0)
    assert M.gini(np.r_[np.zeros(99), 1]) > 0.98


def test_intra_list_diversity():
    X = np.eye(3)
    assert np.isclose(M.intra_list_diversity([np.array([0, 1, 2])], X), 1.0)
    assert np.isclose(M.intra_list_diversity([np.array([0, 0])], X), 0.0)


def test_pareto_mask_definition():
    df = pd.DataFrame({"u": [0.5, 0.5, 0.4, 0.6], "c": [1.0, 2.0, 0.5, 3.0]})
    # fila 1 dominada por fila 0 (misma utilidad, más costo)
    assert list(M.pareto_mask(df, ["u"], ["c"])) == [True, False, True, True]


def test_paired_comparison_counts_discordant_pairs():
    from inf8239_u03.evaluation import paired_comparison
    a = pd.Series([1, 1, 0, 0, 1], index=range(5), dtype=float)
    b = pd.Series([1, 0, 0, 1, 0], index=range(5), dtype=float)
    r = paired_comparison(a, b, n_boot=200)
    assert r["aciertos_solo_A"] == 2 and r["aciertos_solo_B"] == 1
    assert np.isclose(r["dif_HitRate"], 0.2) and r["IC95_inf"] <= 0.2 <= r["IC95_sup"]
