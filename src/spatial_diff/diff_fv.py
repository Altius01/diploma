import numpy as np
import taichi as ti

from src.common.types import *
from src.common.pointers import *


@ti.func
def V_plus_vec_1D(foo: ti.template(), idx):
    result = vec3(0)

    for i in ti.ndrange(2):
        new_idx = idx + vec3i([i, 0, 0])
        result += foo(new_idx)
    return result / 2.0


@ti.func
def V_plus_vec_2D(foo: ti.template(), idx):
    result = vec3(0)

    for i, j in ti.ndrange(2, 2):
        new_idx = idx + vec3i([i, j, 0])
        result += foo(new_idx)
    return result / 4.0


@ti.func
def V_plus_vec_3D(foo: ti.template(), idx):
    result = vec3(0)

    for i, j, k in ti.ndrange(2, 2, 2):
        new_idx = idx + vec3i([i, j, k])
        result += foo(new_idx)
    return result / 8.0


@ti.func
def V_plus_sc_1D(foo: ti.template(), idx):
    result = double(0.0)

    for i in ti.ndrange(2):
        new_idx = idx + vec3i([i, 0, 0])
        result += foo(new_idx)
    return result / 2.0


@ti.func
def V_plus_sc_2D(foo: ti.template(), idx):
    result = double(0.0)

    for i, j in ti.ndrange(2, 2):
        new_idx = idx + vec3i([i, j, 0])
        result += foo(new_idx)
    return result / 4.0


@ti.func
def V_plus_sc_3D(foo: ti.template(), idx):
    result = double(0.0)

    for i, j, k in ti.ndrange(2, 2, 2):
        new_idx = idx + vec3i([i, j, k])
        result += foo(new_idx)
    return result / 8.0


@ti.func
def get_dx_st_1D(diff_axe, i, j, left):
    result = vec3i(0)
    a = 0

    if left == False:
        a = 1

    result = vec3i([a, 0, 0])

    return result


@ti.func
def get_dx_st_2D(diff_axe, i, j, left):
    result = vec3i(0)
    a = 0

    if left == False:
        a = 1

    if diff_axe == 0:
        result = vec3i([a, i, 0])
    elif diff_axe == 1:
        result = vec3i([i, a, 0])

    return result


@ti.func
def get_dx_st_3D(diff_axe, i, j, left):
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
def dx_sc_1D(foo: ti.template(), diff_axe, h, idx):
    result = double(0.0)

    new_idx_l = idx + get_dx_st_1D(diff_axe, 0, 0, True)
    new_idx_r = idx + get_dx_st_1D(diff_axe, 0, 0, False)

    result += (foo(new_idx_r) - foo(new_idx_l)) / h

    return result


@ti.func
def dx_sc_2D(foo: ti.template(), diff_axe, h, idx):
    result = double(0.0)

    for i in ti.ndrange(2):
        new_idx_l = idx + get_dx_st_2D(diff_axe, i, 0, True)
        new_idx_r = idx + get_dx_st_2D(diff_axe, i, 0, False)

        result += (foo(new_idx_r) - foo(new_idx_l)) / h

    return result / 2.0


@ti.func
def dx_sc_3D(foo: ti.template(), diff_axe, h, idx):
    result = double(0.0)

    for i, j in ti.ndrange(2, 2):
        new_idx_l = idx + get_dx_st_3D(diff_axe, i, j, True)
        new_idx_r = idx + get_dx_st_3D(diff_axe, i, j, False)

        result += (foo(new_idx_r) - foo(new_idx_l)) / h

    return result / 4.0


@ti.func
def dx_vec_1D(foo: ti.template(), axe, diff_axe, h, idx):
    result = double(0.0)

    new_idx_l = idx + get_dx_st_1D(diff_axe, 0, 0, True)
    new_idx_r = idx + get_dx_st_1D(diff_axe, 0, 0, False)
    result += (get_elem_1d(foo(new_idx_r), axe) - get_elem_1d(foo(new_idx_l), axe)) / h

    return result


@ti.func
def dx_vec_2D(foo: ti.template(), axe, diff_axe, h, idx):
    result = double(0.0)

    for i in ti.ndrange(2):
        new_idx_l = idx + get_dx_st_2D(diff_axe, i, 0, True)
        new_idx_r = idx + get_dx_st_2D(diff_axe, i, 0, False)
        result += (
            get_elem_1d(foo(new_idx_r), axe) - get_elem_1d(foo(new_idx_l), axe)
        ) / h

    return result / 2.0


@ti.func
def dx_vec_3D(foo: ti.template(), axe, diff_axe, h, idx):
    result = double(0.0)

    for i, j in ti.ndrange(2, 2):
        new_idx_l = idx + get_dx_st_3D(diff_axe, i, j, True)
        new_idx_r = idx + get_dx_st_3D(diff_axe, i, j, False)
        result += (
            get_elem_1d(foo(new_idx_r), axe) - get_elem_1d(foo(new_idx_l), axe)
        ) / h

    return result / 4.0


@ti.func
def grad_sc_1D(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    result[0] = dx_sc_1D(foo, 0, h[0], idx)

    return result


@ti.func
def grad_sc_2D(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    result[0] = dx_sc_2D(foo, 0, h[0], idx)
    result[1] = dx_sc_2D(foo, 1, h[1], idx)

    return result


@ti.func
def grad_sc_3D(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    result[0] = dx_sc_3D(foo, 0, h[0], idx)
    result[1] = dx_sc_3D(foo, 1, h[1], idx)
    result[2] = dx_sc_3D(foo, 2, h[2], idx)

    return result


@ti.func
def grad_vec_1D(foo: ti.template(), h: ti.template(), idx):
    result = mat3x3(0)

    for i, j in ti.ndrange(1, result.m):
        result[i, j] = dx_vec_1D(foo, j, 0, get_elem_1d(h, i), idx)

    return result.transpose()


@ti.func
def grad_vec_2D(foo: ti.template(), h: ti.template(), idx):
    result = mat3x3(0)

    for i, j in ti.ndrange(2, result.m):
        result[i, j] = dx_vec_2D(foo, j, i, get_elem_1d(h, i), idx)

    return result.transpose()


@ti.func
def grad_vec_3D(foo: ti.template(), h: ti.template(), idx):
    result = mat3x3(0)

    for i, j in ti.ndrange(3, result.m):
        result[i, j] = dx_vec_3D(foo, j, i, get_elem_1d(h, i), idx)

    return result.transpose()


@ti.func
def div_vec_1D(foo: ti.template(), h: ti.template(), idx):
    result = double(0.0)

    result += dx_vec_1D(foo, 0, 0, h[0], idx)

    return result


@ti.func
def div_vec_2D(foo: ti.template(), h: ti.template(), idx):
    result = double(0.0)

    for i in ti.ndrange(2):
        result += dx_vec_2D(foo, i, i, get_elem_1d(h, i), idx)

    return result


@ti.func
def div_vec_3D(foo: ti.template(), h: ti.template(), idx):
    result = double(0.0)

    for i in ti.ndrange(vec3.n):
        result += dx_vec_3D(foo, i, i, get_elem_1d(h, i), idx)

    return result


@ti.func
def rot_vec_1D(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    for i, j, k in ti.ndrange(result.n, 1, result.n):
        result[i] += get_elem_3d(levi_chevita, [i, j, k]) * dx_vec_1D(
            foo, k, 0, get_elem_1d(h, j), idx
        )

    return result


@ti.func
def rot_vec_2D(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    for i, j, k in ti.ndrange(result.n, 2, result.n):
        # result[i] += get_elem_3d(levi_chevita, [i, j, k]) * dx_vec_2D(
        #     foo, k, j, get_elem_1d(h, j), idx
        # )
        result[i] += dx_vec_2D(foo, k, j, get_elem_1d(h, j), idx)

    return result


@ti.func
def rot_vec_3D(foo: ti.template(), h: ti.template(), idx):
    result = vec3(0)

    for i, j, k in ti.ndrange(result.n, result.n, result.n):
        result[i] += get_elem_3d(levi_chevita, [i, j, k]) * dx_vec_3D(
            foo, k, j, get_elem_1d(h, j), idx
        )

    return result
