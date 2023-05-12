import pytest

from opta.algorithms.particle_swarm_optimization import PSO
from opta.tools.testing import smoke_check


@pytest.mark.parametrize("_", range(25))
def test_smoke(_):
    algorithm = PSO(25, 0.5, 0.9, 0.9, 0.9)
    assert smoke_check(algorithm, number_of_iterations=10_000)
    try:
        state = algorithm.serialize()
        algorithm.deserialize(state)
        assert True
    except Exception:
        assert False
