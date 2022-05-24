import numpy as np
import numpy.testing as npt
import pytest

from opta.tools.vectors import (
    bound_vector,
    generate_vector_in_area,
    generate_vector_in_sphere,
)


def test_bound_vector_1():
    v = np.array([1, 2, 3])
    bounds = np.array([[-5, 5] * len(v)])
    npt.assert_equal(v, bound_vector(v, bounds))


def test_bound_vector_2():
    v = np.array([1, 2, 3])
    bounds = np.array([[-2, 2] * len(v)])
    npt.assert_equal(np.array([1, 2, 2]), bound_vector(v, bounds))


def test_bound_vector_3():
    v = np.array([5, -5, 0])
    bounds = np.array([[-2, 2] * len(v)])
    npt.assert_equal(np.array([2, -2, 0]), bound_vector(v, bounds))


@pytest.mark.parametrize("_", range(100))
def test_bound_vector_N(_):
    search_area = np.array([[-10, 10], [-10, 10], [-10, 10]])
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    vector = generate_vector_in_area(search_area)
    vector = bound_vector(vector, bounds)
    assert ((bounds[:, 0] <= vector) & (vector <= bounds[:, 1])).all()


@pytest.mark.parametrize("_", range(100))
def test_generate_vector_in_area(_):
    search_area = np.array([[-10, 10], [-10, 10]])
    vector = generate_vector_in_area(search_area)
    assert ((search_area[:, 0] <= vector) & (vector <= search_area[:, 1])).all()


@pytest.mark.parametrize("_", range(100))
def test_generate_vector_in_sphere(_):
    initial = np.array([0, 0, 0])
    radius = 2.5
    generated = generate_vector_in_sphere(initial, radius)
    assert np.linalg.norm(generated - initial) <= radius
