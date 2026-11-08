import pytest

from inf8239_u03.base import ItemIndex
from inf8239_u03.content import ContentRecommender
from inf8239_u03.data import make_synthetic_movielens
from inf8239_u03.split import leave_last_out


@pytest.fixture(scope="session")
def ml():
    return make_synthetic_movielens(n_users=80, n_items=150, seed=11)


@pytest.fixture(scope="session")
def index(ml):
    return ItemIndex(ml.movies)


@pytest.fixture(scope="session")
def split(ml):
    return leave_last_out(ml.ratings)


@pytest.fixture(scope="session")
def content(ml, index):
    return ContentRecommender().fit(None, index, ml.movies)
