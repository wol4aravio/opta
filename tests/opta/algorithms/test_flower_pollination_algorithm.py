import pytest

from opta.algorithms.flower_pollination_algorithm import FPA
from opta.tools.testing import smoke_check


@pytest.mark.parametrize("_", range(25))
def test_smoke(_):
    algorithm = FPA(25, 0.9, 0.25, 1.25)
    assert smoke_check(algorithm, number_of_iterations=10_000)
    try:
        state = algorithm.serialize()
        algorithm.deserialize(state)
        assert True
    except Exception:
        assert False
