import zipfile

import pandas as pd
import pytest

from inf8239_u03.data import MovieLens, audit, install_zip, load_movielens
from inf8239_u03.split import cold_user_split, leave_last_out, merge_train_valid


def test_audit_passes_on_valid_schema(ml):
    res = audit(ml)
    assert res["usuarios"] == 80 and res["claves_foraneas_ok"]
    assert 0 < res["densidad"] < 1


def test_audit_detects_bad_rating(ml):
    bad = ml.ratings.copy()
    bad.loc[0, "rating"] = 7.0
    with pytest.raises(AssertionError, match="rating"):
        audit(MovieLens(bad, ml.movies))


def test_audit_detects_broken_foreign_key(ml):
    bad = ml.ratings.copy()
    bad.loc[0, "movieId"] = -1
    with pytest.raises(AssertionError, match="movieId"):
        audit(MovieLens(bad, ml.movies))


def test_install_zip_records_hash(tmp_path, ml):
    src = tmp_path / "ml-latest-small"
    src.mkdir()
    ml.ratings.to_csv(src / "ratings.csv", index=False)
    ml.movies.to_csv(src / "movies.csv", index=False)
    z = tmp_path / "ml.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in ["ratings.csv", "movies.csv"]:
            zf.write(src / f, f"ml-latest-small/{f}")
    man = install_zip(z, source="test", dest=tmp_path / "raw")
    assert len(man["sha256_zip"]) == 64
    assert (tmp_path / "raw" / "download_manifest.json").exists()
    back = load_movielens(tmp_path / "raw" / "ml-latest-small")
    pd.testing.assert_frame_equal(back.ratings, ml.ratings)


def test_temporal_split_has_no_future_leakage(split):
    tmax_train = split.train.groupby("userId").timestamp.max()
    t_valid = split.valid.set_index("userId").timestamp
    t_test = split.test.set_index("userId").timestamp
    assert (t_valid >= tmax_train.reindex(t_valid.index)).all()
    assert (t_test >= t_valid.reindex(t_test.index)).all()


def test_split_is_a_partition(ml, split):
    total = len(split.train) + len(split.valid) + len(split.test)
    assert total == len(ml.ratings)
    assert split.test.userId.is_unique and split.valid.userId.is_unique
    assert len(merge_train_valid(split)) == len(split.train) + len(split.valid)


def test_cold_users_absent_from_train(ml):
    cs = cold_user_split(ml.ratings, frac=0.2, n_onboarding=3, seed=0)
    assert not cs.train.userId.isin(cs.cold_users).any()
    assert cs.onboarding.groupby("userId").size().max() <= 3
    assert set(cs.test.userId) == set(cs.cold_users)
    assert cs.test.userId.is_unique
    # el onboarding es anterior a la interacción evaluada
    m = cs.onboarding.groupby("userId").timestamp.max()
    assert (cs.test.set_index("userId").timestamp.reindex(m.index) >= m).all()


def test_cold_split_can_exclude_users(ml):
    a = cold_user_split(ml.ratings, frac=0.2, seed=0)
    b = cold_user_split(ml.ratings, frac=0.2, seed=1, exclude=a.cold_users)
    assert not set(a.cold_users) & set(b.cold_users)
    assert len(a.cold_users) == len(b.cold_users)
