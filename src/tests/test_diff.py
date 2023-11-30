import pytest
import numpy as np
import taichi as ti
from src.spatial_diff.diff_fv import grad_sc_3D

from src.common.types import *


@ti.func
def _check_ghost(idx, shape, ghosts):
    return (idx < ghosts) or (idx >= shape + ghosts)


@ti.func
def check_ghost_idx(idx, N, ghost):
    result = False

    for i in ti.ndrange(3):
        result = result or _check_ghost(idx[i], N, ghost)

    return result


@ti.kernel
def compute_gradient(
    field: ti.template(), grad: ti.template(), h: double, N: int, ghost: int
):
    for idx in ti.grouped(ti.ndrange(N + 2 * ghost, N + 2 * ghost, N + 2 * ghost)):
        if not check_ghost_idx(idx, N, ghost):
            grad[idx] = grad_sc_3D(foo=field, h=vec3(h), idx=idx)


def test_grad_scalar_field(
    rosenbrock,
    grad_rosenbrock,
    scalar_field,
    N,
    N_ghsot,
):
    step = 2 * np.pi / N
    x0 = np.mgrid[
        -N_ghsot * step : (2 * np.pi + N_ghsot * step) : step,
        -N_ghsot * step : (2 * np.pi + N_ghsot * step) : step,
        -N_ghsot * step : (2 * np.pi + N_ghsot * step) : step,
    ]
    x1 = np.mgrid[0 : 2 * np.pi : step, 0 : 2 * np.pi : step, 0 : 2 * np.pi : step]

    field = rosenbrock(x=x0)

    grad_field = np.moveaxis(grad_rosenbrock(x=x1), 0, -1)

    ti_field = ti.field(double, shape=field.shape)

    ti_field.from_numpy(field)

    ti_grad = ti.Vector.field(n=3, dtype=double, shape=(N, N, N))

    print(ti_field.shape, ti_grad.shape)

    @ti.func
    def get_field(idx):
        return ti_field[idx]

    compute_gradient(get_field, ti_grad, step, N, N_ghsot)

    ti_grad = ti_grad.to_numpy()

    print(np.max(np.abs(field)))

    print(np.max(np.abs(ti_grad)), np.max(np.abs(grad_field)))
    assert np.allclose(grad_field, ti_grad)