import pytest

from opta.algorithms.black_hole_optimization import BHO
from opta.tools.testing import smoke_check


@pytest.mark.parametrize("_", range(10))
def test_smoke(_):
    algorithm = BHO(100)
    assert smoke_check(algorithm, number_of_iterations=1_000)
    try:
        state = algorithm.serialize()
        algorithm.deserialize(state)
        assert True
    except Exception:
        assert False
