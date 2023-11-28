import pytest
import taichi as ti
import numpy as np

from src.common.types import *

__N = 6
__N_ghost = 3

__a = 2
__b = 5
__c = 8


def _rosenbrock(x):
    """
    f(x, y, z) = (a - x)^2 + b(y - x^2)^2 + c(z - x^3)^2
    :param x: is Ndarrary of shape (3, N, N, N)
    :return: NDarray f(x, y, z)
    """
    return (
        np.power(__a - x[0], 2)
        + __b * np.power((x[1] - np.power(x[0], 2)), 2)
        + __c * np.power((x[2] - np.power(x[0], 3)), 2)
    )


def _grad_rosenbrock(x):
    return np.array(
        [
            -2.0 * __a
            + 2.0
            * x[0]
            * (
                1.0
                + 3.0 * __c * np.power(x[0], 4)
                + 2.0 * __b * (np.power(x[0], 2) - x[1])
                - 3.0 * __c * x[0] * x[2]
            ),
            2.0 * __b * (-np.power(x[0], 2) + x[1]),
            2.0 * __c * (-np.power(x[0], 3) + x[2]),
        ]
    )


@pytest.fixture(scope="session")
def rosenbrock():
    return _rosenbrock


@pytest.fixture(scope="session")
def grad_rosenbrock():
    return _grad_rosenbrock


@pytest.fixture(scope="session")
def init_taichi():
    ti.init(ti.cpu)


@pytest.fixture(scope="session")
def N(init_taichi):
    return __N


@pytest.fixture(scope="session")
def N_ghsot(init_taichi):
    return __N_ghost


@pytest.fixture()
def scalar_field(N):
    return ti.field(dtype=double, shape=(N, N, N))


@pytest.fixture()
def vector3_field(shape):
    return ti.field(n=3, dtype=double, shape=shape)


@pytest.fixture()
def mat3x3_field_2d(shape):
    return ti.field(n=3, m=3, dtype=double, shape=shape)
