import numpy as np

from inf8239_u03.base import History, top_k
from inf8239_u03.popularity import PopularitySmoothed, popularity_table


def test_similar_items_excludes_query(content, index):
    mid = int(index.movie_ids[0])
    recs = content.similar_items(mid, k=10)
    assert mid not in recs.movieId.values
    assert len(recs) == 10
    assert recs.similarity.is_monotonic_decreasing


def test_tfidf_rows_are_unit_norm(content):
    norms = np.linalg.norm(content.X, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)


def test_identical_genres_have_similarity_one(content, index):
    g = index.genres
    for i in range(len(g)):
        j = np.where(g == g[i])[0]
        if len(j) > 1:
            assert np.isclose(content.X[j[0]] @ content.X[j[1]], 1.0)
            return


def test_user_profile_follows_liked_genres(content):
    liked, disliked = 0, 1
    h = History(np.array([liked, disliked]), np.array([5.0, 1.0]))
    s = content.score(1, h)
    assert s[liked] > s[disliked]


def test_smoothed_popularity_shrinks_single_votes(ml):
    r = ml.ratings.copy()
    lone = ml.movies.movieId.iloc[-1]
    r = r[r.movieId != lone]
    r.loc[len(r)] = [9999, lone, 5.0, 2_000_000_000]
    tab = popularity_table(r, ml.movies)
    row = tab[tab.movieId == lone].iloc[0]
    assert row["mean"] == 5.0 and row["weighted_score"] < 4.5
    assert tab.weighted_score.iloc[0] >= row["weighted_score"]


def test_top_k_excludes_seen_and_breaks_ties():
    s = np.array([1.0, 1.0, 0.5, 1.0])
    tb = np.array([0, 5, 0, 9])
    out = top_k(s, 3, exclude=np.array([3]), tiebreak=tb)
    assert list(out) == [1, 0, 2]


def test_popularity_predict_unknown_item_returns_global_mean(split, index):
    pop = PopularitySmoothed().fit(split.train, index)
    assert np.isclose(pop.predict([1], [-12345])[0], split.train.rating.mean())
