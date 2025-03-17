import pytest

from opta.algorithms.random_search import RS
from opta.tools.testing import smoke_check


@pytest.mark.parametrize("_", range(5))
def test_smoke(_):
    algorithm = RS(1e-1)
    assert smoke_check(algorithm, number_of_iterations=10_000)
    try:
        state = algorithm.serialize()
        algorithm.deserialize(state)
        assert True
    except Exception:
        assert False
