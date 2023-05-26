import numpy as np
import taichi as ti

from taichi_src.kernels.common.types import *

# # type hints
# # double = ti.types.f64
# double = ti.types.f32

# vec3 = ti.types.vector(n=3, dtype=double)
# vec4 = ti.types.vector(n=4, dtype=double)
# vec5 = ti.types.vector(n=5, dtype=double)
# mat3x3 = ti.types.matrix(n=3, m=3, dtype=double)

# vec0i = ti.types.vector(n=0, dtype=int)
# vec1i = ti.types.vector(n=1, dtype=int)
# vec2i = ti.types.vector(n=2, dtype=int)
# vec3i = ti.types.vector(n=3, dtype=int)
# mat5x2i = ti.types.matrix(5, 2, int)
# mat5x3i = ti.types.matrix(5, 3, int)
# # type hints

# kron = mat3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# levi_chevita = np.array([
#     [[0, 0, 0],
#      [0, 0, 1],
#      [0, -1, 0]],
#     [[0, 0, -1],
#      [0, 0, 0],
#      [1, 0, 0]],
#     [[0, 1, 0],
#      [-1, 0, 0],
#      [0, 0, 0]],
# ], dtype=np.float64)

# @ti.func
# def get_elem(arr: ti.template(), idx: int):
#     result = arr[0]
#     if idx == 1:
#         result = arr[1]
#     elif idx == 2:
#         result = arr[2]

#     return result
        

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
        coefs = vec5([1.0/12, -8.0/12, 0.0, 8.0/12, -1.0/12])
    elif order == 2:
        if axe_1 == axe_2:
            coefs = vec5([-1.0/12, 16.0/12, -30.0/12, 16.0/12, -1.0/12])
        else:
            coefs = vec5([-1.0/4, -1.0/4, 0, -1.0/4, 1.0/4])

    return coefs

@ti.func
def get_weighted_sum(foo: ti.template(), axes: ti.template(), stencil: ti.template(), coefs: ti.template()):
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
def dx(foo: ti.template(), axes, diff_axe: ti.i32, h:double, idx):
    coefs = get_diff_coefs(diff_axe, diff_axe, 1) / h
    stencil = get_diff_stencil(diff_axe, diff_axe, idx, 1)

    return get_weighted_sum(foo, axes, stencil, coefs)

@ti.func
def ddx(foo: ti.template(), axes, diff_axe_1: ti.i32, diff_axe_2: ti.i32, h1: double, h2: double, idx):
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