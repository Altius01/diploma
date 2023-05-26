import taichi as ti

from taichi_src.common.types import *

def static_get_shape(a):
    try:
        return len(a.get_shape())
    except:
        return 0


def static_get_len(a):
    try:
        return len(a)
    except:
        return 1


@ti.func
def norm_dot(a: ti.template(), b: ti.template()):
    len_shape = ti.static(static_get_shape(a))

    result = double(0.0)
    if ti.static(len_shape == 0):
        result = ti.sqrt(a * b)
    elif ti.static(len_shape == 1):
        result = (a * b).norm()
    elif ti.static(len_shape == 2):
        for i, j in ti.static(ti.ndrange(a.n, a.m)):
            result += a[i, j] * b[i, j]
        result = ti.sqrt(result)
    
    return result


@ti.func
def hadamar_dot(a: ti.template(), b: ti.template()):
    len_shape = ti.static(static_get_shape(a))

    result = a
    if ti.static(len_shape in [0, 1]):
        result = a * b
    elif ti.static(len_shape == 2):
        for i, j in ti.static(ti.ndrange(a.n, a.m)):
            result[i, j] = a[i, j] * b[i, j]
    
    return result


@ti.func
def trace_sqr(arr: mat3x3):
    result = double(0.0)
    for i in ti.static(range(3)):
        result += arr[i, i]**2

    return result


@ti.func
def get_mat_row(arr: ti.template(), idx: int):
    shape = ti.static(arr.get_shape())

    if ti.static(len(shape) == 1):
        result = arr

        return result
    elif ti.static(len(shape) == 2):
        result = arr[0, :]
        if idx == 1:
            result = arr[1, :]
        elif idx == 2:
            result = arr[2, :]

        return result


@ti.func
def get_mat_col(arr: ti.template(), idx: int):
    shape = ti.static(arr.get_shape())

    if ti.static(len(shape) == 1):
        result = arr[0]
        if idx == 1:
            result = arr[1]
        elif idx == 2:
            result = arr[2]
        
        return result
    elif ti.static(len(shape) == 2):
        result = arr[:, 0]
        if idx == 1:
            result = arr[:, 1]
        elif idx == 2:
            result = arr[:, 2]

        return result
    return 0


@ti.func
def get_basis(j: int):
    result = vec3i(0)

    if j == 0:
        result[0] = 1
    elif j == 1:
        result[1] = 1
    elif j == 2:
        result[2] = 1

    return result
