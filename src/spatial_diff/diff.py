import numpy as np
import taichi as ti
from src.common.pointers import get_elem

from src.common.types import *
import src.common.matrix_ops as mat_ops


@ti.func
def get_diff_stencil(axe_1, axe_2, idx, order):
    result = mat5x3i([idx, idx, idx, idx, idx])
    stencil = mat5x2i(0)

    if order == 1:
        stencil = mat5x2i([(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)])
    elif order == 2:
        if axe_1 == axe_2:
            stencil = mat5x2i([(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)])
        else:
            stencil = mat5x2i([(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)])

    for i in range(5):
        _idx = idx
        if axe_1 == 0:
            _idx[0] += stencil[i, 0]
        elif axe_1 == 1:
            _idx[1] += stencil[i, 0]
        elif axe_1 == 2:
            _idx[2] += stencil[i, 0]

        if axe_2 == 0:
            _idx[0] += stencil[i, 1]
        elif axe_2 == 1:
            _idx[1] += stencil[i, 1]
        elif axe_2 == 2:
            _idx[2] += stencil[i, 1]

        for j in range(3):
            result[i, j] = _idx[j]

    return result


@ti.func
def get_diff_coefs(axe_1, axe_2, order):
    coefs = vec5(0)

    if order == 1:
        coefs = vec5([1.0 / 12, -8.0 / 12, 0.0, 8.0 / 12, -1.0 / 12])
    elif order == 2:
        if axe_1 == axe_2:
            coefs = vec5([-1.0 / 12, 16.0 / 12, -30.0 / 12, 16.0 / 12, -1.0 / 12])
        else:
            coefs = vec5([-1.0 / 4, -1.0 / 4, 0, -1.0 / 4, 1.0 / 4])

    return coefs


@ti.func
def get_weno(foo: ti.template(), i, idx):
    fm2 = foo(idx - 2 * mat_ops.get_basis(i))
    fm1 = foo(idx - mat_ops.get_basis(i))
    f = foo(idx)
    fp1 = foo(idx + mat_ops.get_basis(i))
    fp2 = foo(idx + 2 * mat_ops.get_basis(i))

    g1 = 0.1
    g2 = 0.6
    g3 = 0.3

    b1 = (13.0 / 12.0) * (fm2 - 2 * fm1 + f) ** 2 + 0.25 * (fm2 - 4 * fm1 + 3 * f) ** 2
    b2 = (13.0 / 12.0) * (fm1 - 2 * f + fp1) ** 2 + 0.25 * (fm1 - fp1) ** 2
    b3 = (13.0 / 12.0) * (f - 2 * fp1 + fp2) ** 2 + 0.25 * (3 * f - 4 * fp1 + fp2) ** 2

    a1 = g1 / (b1 + 1e-6) ** 2
    a2 = g2 / (b2 + 1e-6) ** 2
    a3 = g3 / (b3 + 1e-6) ** 2

    a_sum = a1 + a2 + a3

    w1 = a1 / a_sum
    w2 = a1 / a_sum
    w3 = a1 / a_sum

    return (
        w1 * ((1.0 / 3.0) * fm2 - (7.0 / 6.0) * fm1 + (11.0 / 6.0) * f)
        + w2 * (-(1.0 / 6.0) * fm1 + (5.0 / 6.0) * f + (1.0 / 3.0) * fp1)
        + w3 * ((1.0 / 3.0) * f + (5.0 / 6.0) * fp1 - (1.0 / 6.0) * fp2)
    )


@ti.func
def get_weighted_sum(
    foo: ti.template(),
    axes: ti.template(),
    stencil: ti.template(),
    coefs: ti.template(),
):
    result = double(0.0)
    for i in range(ti.static(len(stencil))):
        idx = vec3i(0)
        for j in range(3):
            idx[j] = stencil[i, j]

        if ti.static(axes.n == 1):
            result += foo(idx)[axes[0]] * coefs[i]
        elif ti.static(axes.n == 2):
            mat_ = foo(idx)
            result += mat_[axes[0], axes[1]] * coefs[i]
        else:
            result += foo(idx) * coefs[i]

    return result


@ti.func
def get_weighted_sum_sc(
    foo: ti.template(), stencil: ti.template(), coefs: ti.template()
):
    result = double(0.0)
    for i in ti.ndrange(5):
        idx = vec3i(0)
        for j in ti.ndrange(3):
            idx[j] = stencil[i, j]

        result += foo(idx) * coefs[i]

    return result


@ti.func
def get_weighted_sum_vec(
    foo: ti.template(), axe: ti.template(), stencil: ti.template(), coefs: ti.template()
):
    result = double(0.0)
    for i in ti.ndrange(5):
        idx = vec3i(0)
        for j in ti.ndrange(3):
            idx[j] = stencil[i, j]

        result += foo(idx)[axe] * coefs[i]

    return result


@ti.func
def dx_sc(foo: ti.template(), diff_axe: ti.i32, h: double, idx):
    coefs = get_diff_coefs(diff_axe, diff_axe, 1) / h
    stencil = get_diff_stencil(diff_axe, diff_axe, idx, 1)

    return get_weighted_sum_sc(foo, stencil, coefs)


@ti.func
def dx_vec(foo: ti.template(), axes, diff_axe: ti.i32, h: double, idx):
    coefs = get_diff_coefs(diff_axe, diff_axe, 1) / h
    stencil = get_diff_stencil(diff_axe, diff_axe, idx, 1)

    return get_weighted_sum_vec(foo, axes, stencil, coefs)


@ti.func
def dx(foo: ti.template(), axes, diff_axe: ti.i32, h: double, idx):
    coefs = get_diff_coefs(diff_axe, diff_axe, 1) / h
    stencil = get_diff_stencil(diff_axe, diff_axe, idx, 1)

    return get_weighted_sum(foo, axes, stencil, coefs)


@ti.func
def ddx(
    foo: ti.template(),
    axes,
    diff_axe_1: ti.i32,
    diff_axe_2: ti.i32,
    h1: double,
    h2: double,
    idx,
):
    coefs = get_diff_coefs(diff_axe_1, diff_axe_2, 2) / h1 * h2
    stencil = get_diff_stencil(diff_axe_1, diff_axe_2, idx, 2)

    return get_weighted_sum(foo, axes, stencil, coefs)


@ti.func
def grad(foo: ti.template(), h, idx):
    result = vec3(0)

    for i in range(3):
        result[i] = dx(foo, vec0i(0), i, get_elem(h, i), idx)
    return result


@ti.func
def div_vec3(foo: ti.template(), h, idx):
    result = double(0.0)

    for i in range(3):
        result += dx(foo, vec1i(i), i, get_elem(h, i), idx)
    return result


@ti.func
def rot_vec3(foo: ti.template(), h, idx):
    result = vec3(0.0)

    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
        result[i] += levi_chevita[i][j][k] * dx(foo, vec1i(k), j, get_elem(h, j), idx)
    return result


@ti.func
def div_mat3x3(foo: ti.template(), axe, h, idx):
    result = vec3(0)
    axes = vec2i(0)
    for i, j in ti.ndrange(3, 3):
        if axe == 0:
            axes = vec2i([j, i])
        elif axe == 1:
            axes = vec2i([i, j])

        result[i] += dx(foo, axes, j, get_elem(h, j), idx)
    return result
