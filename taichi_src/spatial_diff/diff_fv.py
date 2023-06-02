import numpy as np
import taichi as ti

from taichi_src.common.types import *
from taichi_src.common.pointers import *

import taichi_src.spatial_diff.diff as diff_fd

@ti.func
def V_plus_vec(foo: ti.template(), idx):
    result = vec3(0)

    for i, j, k in ti.ndrange(2, 2, 2):
        new_idx = idx + vec3i([i, j, k])
        result += foo(new_idx)
    return result / 8.0

@ti.func
def V_plus_sc(foo: ti.template(), idx):
    result = double(0.0)

    for i, j, k in ti.ndrange(2, 2, 2):
        new_idx = idx + vec3i([i, j, k])
        result += foo(new_idx)
    return result / 8.0

@ti.func
def get_dx_st(diff_axe, i, j, left=True):
    result = vec3i(0)
    a = 0

    if left == False:
        a = 1

    if diff_axe == 0:
        result = vec3i([a, i, j])
    elif diff_axe == 1:
        result = vec3i([i, a, j])
    elif diff_axe == 2:
        result = vec3i([i, j, a])

    return result

@ti.func
def dx_sc(foo: ti.template(), diff_axe, h, idx):
    result = double(0.0)

    for i, j in (ti.ndrange(2, 2)):
        new_idx_l = idx + get_dx_st(diff_axe, i, j, left=True)
        new_idx_r = idx + get_dx_st(diff_axe, i, j, left=False)

        result += ( foo(new_idx_r) - foo(new_idx_l) ) / h

        # dx_idx = idx + get_dx_st(diff_axe, i, j, left=True)
        # result += diff_fd.dx_sc(foo, diff_axe, h, dx_idx)

    return result / 4.0

@ti.func
def dx_vec(foo: ti.template(), axe, diff_axe, h, idx):
    result = double(0.0)

    for i, j in (ti.ndrange(2, 2)):
        new_idx_l = idx + get_dx_st(diff_axe, i, j, left=True)
        new_idx_r = idx + get_dx_st(diff_axe, i, j, left=False)
        result += ( get_elem_1d(foo(new_idx_r), axe) 
            - get_elem_1d(foo(new_idx_l), axe) ) / h

        # dx_idx = idx + get_dx_st(diff_axe, i, j, left=True)
        # result += diff_fd.dx_vec(foo, axe, diff_axe, h, dx_idx)

    return result / 4.0

@ti.func
def grad_sc(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    result[0] = dx_sc(foo, 0, h[0], idx)
    result[1] = dx_sc(foo, 1, h[1], idx)
    result[2] = dx_sc(foo, 2, h[2], idx)

    return result

@ti.func
def grad_vec(foo: ti.template(), h: ti.template(), idx):
    result = mat3x3(0)

    for i, j in ti.static(ti.ndrange(result.n, result.m)):
        result[i, j] = dx_vec(foo, j, i, h[i], idx)

    return result

@ti.func
def div_vec(foo: ti.template(), h: ti.template(), idx):
    result = double(0.0)

    for i in ti.static(ti.ndrange(vec3.n)):
        result += dx_vec(foo, i, i, h[i], idx)

    return result

@ti.func
def rot_vec(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    for i, j, k in (ti.ndrange(result.n, result.n, result.n)):
        result[i] += get_elem_3d(levi_chevita, [i, j, k]) * dx_vec(foo, k, j, get_elem_1d(h, j), idx)

    return result
