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
def norm_dot_sc(a: ti.template(), b: ti.template()):
    result = ti.sqrt(ti.abs(a*b))

    return result

@ti.func
def norm_dot_vec(a: ti.template(), b: ti.template()):
    result = (a * b).norm()
    
    return result

@ti.func
def norm_dot_mat(a: ti.template(), b: ti.template()):
    result = double(0.0)
    for i, j in ti.ndrange(a.n, a.m):
        result += a[i, j] * b[i, j]
    
    result = ti.sqrt(ti.abs(result))
    return result

@ti.func
def norm_sqr_dot_mat(a: ti.template(), b: ti.template()):
    result = double(0.0)
    for i, j in ti.ndrange(a.n, a.m):
        result += a[i, j] * b[i, j]
    
    return result


@ti.func
def hadamar_dot_vec(a: ti.template(), b: ti.template()):
    return a * b

@ti.func
def hadamar_dot_mat(a: ti.template(), b: ti.template()):
    result = a

    for i, j in ti.ndrange(a.n, a.m):
        result[i, j] = a[i, j] * b[i, j]
    
    return result


@ti.func
def trace_sqr(arr: mat3x3):
    result = double(0.0)
    for i in ti.ndrange(3):
        result += arr[i, i]**2

    return result

@ti.func
def trace(arr: mat3x3):
    result = double(0.0)
    for i in ti.ndrange(3):
        result += arr[i, i]

    return result
    

@ti.func
def get_mat_row(arr: ti.template(), idx: int):
    result = arr[0, :]
    if idx == 1:
        result = arr[1, :]
    elif idx == 2:
        result = arr[2, :]

    return result


@ti.func
def get_mat_col(arr: ti.template(), idx: int):
    result = arr[:, 0]
    if idx == 1:
        result = arr[:, 1]
    elif idx == 2:
        result = arr[:, 2]

    return result

@ti.func
def get_vec_col(arr: ti.template(), idx: int):
    result = arr[0]
    if idx == 1:
        result = arr[1]
    elif idx == 2:
        result = arr[2]
    
    return result


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
