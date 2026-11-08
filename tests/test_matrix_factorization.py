import numpy as np

from inf8239_u03.base import History, histories
from inf8239_u03.evaluation import evaluate_ratings
from inf8239_u03.hybrid import HybridRecommender
from inf8239_u03.matrix_factorization import BiasBaseline, MatrixFactorization
from inf8239_u03.popularity import GlobalMean, PopularityCount


def test_mf_training_error_decreases(split, index):
    mf = MatrixFactorization(factors=8, epochs=6, seed=0).fit(split.train, index)
    assert mf.history_[-1] < mf.history_[0]


def test_mf_is_deterministic_with_seed(split, index):
    a = MatrixFactorization(factors=5, epochs=2, seed=3).fit(split.train, index)
    b = MatrixFactorization(factors=5, epochs=2, seed=3).fit(split.train, index)
    assert np.allclose(a.P, b.P) and np.allclose(a.Q, b.Q)


def test_mf_beats_global_mean_on_validation(split, index):
    mf = MatrixFactorization(factors=8, epochs=15, seed=0).fit(split.train, index)
    gm = GlobalMean().fit(split.train, index)
    assert evaluate_ratings(mf, split.valid)["RMSE"] < evaluate_ratings(gm, split.valid)["RMSE"]


def test_bias_baseline_predictions_in_range(split, index):
    b = BiasBaseline().fit(split.train, index)
    p = b.predict(split.valid.userId, split.valid.movieId)
    assert p.min() >= 0.5 and p.max() <= 5.0


def test_mf_unknown_user_has_no_signal(split, index):
    mf = MatrixFactorization(factors=4, epochs=1).fit(split.train, index)
    assert not mf.knows_user(10**9)
    assert np.isnan(mf.score(10**9, History())).all()
    assert mf.n_params() == mf.P.size + mf.Q.size + mf.bu.size + mf.bi.size + 1


def test_recommendations_exclude_seen_items(split, index, content):
    mf = MatrixFactorization(factors=4, epochs=2).fit(split.train, index)
    pop = PopularityCount().fit(split.train, index)
    h = histories(split.train, index)
    hyb = HybridRecommender(mf, content, pop, alpha=0.75)
    for uid in list(h)[:20]:
        recs = hyb.recommend(uid, h[uid], k=10)
        assert len(recs) == 10 and not set(recs) & set(h[uid].items)
        assert hyb.last_source == "híbrido"


def test_hybrid_alpha_extremes_match_components(split, index, content):
    mf = MatrixFactorization(factors=4, epochs=3, seed=1).fit(split.train, index)
    pop = PopularityCount().fit(split.train, index)
    h = histories(split.train, index)
    uid = next(iter(h))
    only_cf = HybridRecommender(mf, content, pop, alpha=1.0).recommend(uid, h[uid], 10)
    assert list(only_cf) == list(mf.recommend(uid, h[uid], 10))
    s0 = HybridRecommender(mf, content, pop, alpha=0.0).score(uid, h[uid])
    assert np.argmax(s0) == np.argmax(content.score(uid, h[uid]))


def test_cold_user_fallback_is_declared(split, index, content):
    mf = MatrixFactorization(factors=4, epochs=1).fit(split.train, index)
    pop = PopularityCount().fit(split.train, index)
    hyb = HybridRecommender(mf, content, pop)
    recs = hyb.recommend(None, History(), k=10)
    assert len(recs) == 10 and hyb.last_source.startswith("popularidad") and not hyb.personalized
    onb = History(np.array([0, 1, 2]), np.array([5.0, 4.0, 2.0]))
    recs = hyb.recommend(10**9, onb, k=10)
    assert hyb.last_source == "contenido+popularidad" and not set(recs) & {0, 1, 2}


def test_gamma_one_reduces_to_popularity(split, index, content):
    mf = MatrixFactorization(factors=4, epochs=2, seed=1).fit(split.train, index)
    pop = PopularityCount().fit(split.train, index)
    h = histories(split.train, index)
    uid = next(iter(h))
    a = HybridRecommender(mf, content, pop, alpha=0.5, gamma=1.0).recommend(uid, h[uid], 10, tiebreak=pop.counts)
    b = pop.recommend(uid, h[uid], 10, tiebreak=pop.counts)
    assert list(a) == list(b)


def test_hybrid_rejects_out_of_range_weights(split, index, content):
    import pytest
    mf = MatrixFactorization(factors=2, epochs=0).fit(split.train, index)
    with pytest.raises(ValueError):
        HybridRecommender(mf, content, PopularityCount(), gamma=1.5)
