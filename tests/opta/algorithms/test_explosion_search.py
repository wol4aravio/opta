import pytest

from opta.algorithms.explosion_search import ES
from opta.tools.testing import smoke_check


@pytest.mark.parametrize("_", range(25))
def test_smoke(_):
    algorithm = ES(6, 1.0)
    assert smoke_check(algorithm, number_of_iterations=10_000)
    try:
        state = algorithm.serialize()
        algorithm.deserialize(state)
        assert True
    except Exception:
        assert False
