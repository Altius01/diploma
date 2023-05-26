import numpy as np
import taichi as ti

from taichi_src.kernels.common.types import *

ti.init()

def static_get_shape(a):
    try:
        return len(a.get_shape())
    except:
        return 0

def static_get_len(a):
    try:
        return len(a)
    except:
        return 0

@ti.func
def get_elem_1d(arr: ti.template(), idx):
    result = arr[0]
    if idx == 1:
        result = arr[1]
    elif idx == 2:
        result = arr[2]

    return result

@ti.func
def get_elem_2d(arr: ti.template(), idx):
    result = arr[0, 0]
    if idx[0] == 0:
        if idx[1] == 0:
            arr[0, 0]
        elif idx[1] == 1:
            arr[0, 1]
        elif idx[1] == 2:
            arr[0, 2]
    elif idx[0] == 1:
        if idx[1] == 0:
            arr[1, 0]
        elif idx[1] == 1:
            arr[1, 1]
        elif idx[1] == 2:
            arr[1, 2]
    elif idx[0] == 2:
        if idx[1] == 0:
            arr[2, 0]
        elif idx[1] == 1:
            arr[2, 1]
        elif idx[1] == 2:
            arr[2, 2]

    return result

@ti.func
def get_elem_3d(arr: ti.template(), idx):
    result = arr[0, 0, 0]
    if idx[0] == 0:
        if idx[1] == 0:
            if idx[2] == 0:
                result = arr[0, 0, 0]
            elif idx[2] == 1:
                result = arr[0, 0, 1]
            elif idx[2] == 2:
                result = arr[0, 0, 2]
        elif idx[1] == 1:
            if idx[2] == 0:
                result = arr[0, 1, 0]
            elif idx[2] == 1:
                result = arr[0, 1, 1]
            elif idx[2] == 2:
                result = arr[0, 1, 2]
        elif idx[1] == 2:
            if idx[2] == 0:
                result = arr[0, 2, 0]
            elif idx[2] == 1:
                result = arr[0, 2, 1]
            elif idx[2] == 2:
                result = arr[0, 2, 2]
    elif idx[0] == 1:
        if idx[1] == 0:
            if idx[2] == 0:
                result = arr[1, 0, 0]
            elif idx[2] == 1:
                result = arr[1, 0, 1]
            elif idx[2] == 2:
                result = arr[1, 0, 2]
        elif idx[1] == 1:
            if idx[2] == 0:
                result = arr[1, 1, 0]
            elif idx[2] == 1:
                result = arr[1, 1, 1]
            elif idx[2] == 2:
                result = arr[1, 1, 2]
        elif idx[1] == 2:
            if idx[2] == 0:
                result = arr[1, 2, 0]
            elif idx[2] == 1:
                result = arr[1, 2, 1]
            elif idx[2] == 2:
                result = arr[1, 2, 2]
    elif idx[0] == 2:
        if idx[1] == 0:
            if idx[2] == 0:
                result = arr[2, 0, 0]
            elif idx[2] == 1:
                result = arr[2, 0, 1]
            elif idx[2] == 2:
                result = arr[2, 0, 2]
        elif idx[1] == 1:
            if idx[2] == 0:
                result = arr[2, 1, 0]
            elif idx[2] == 1:
                result = arr[2, 1, 1]
            elif idx[2] == 2:
                result = arr[2, 1, 2]
        elif idx[1] == 2:
            if idx[2] == 0:
                result = arr[2, 2, 0]
            elif idx[2] == 1:
                result = arr[2, 2, 1]
            elif idx[2] == 2:
                result = arr[2, 2, 2]

    return result

@ti.func
def get_elem(arr: ti.template(), idx):
    idx_len = ti.static(static_get_len(idx))

    if ti.static(idx_len==0):
        return get_elem_1d(arr, idx)
    elif ti.static(idx_len==2):
        return get_elem_2d(arr, idx)
    elif ti.static(idx_len==3):
        return get_elem_3d(arr, idx)

@ti.func
def grad_vec(foo: ti.template(), h: ti.template()):
    result = mat3x3(0)

    for i, j in ti.ndrange(result.n, result.m):
        result[i, j] = foo[j] * h[i]

    return result

@ti.kernel
def test():
    a = vec3([1, 2, 3])
    b = vec3([4, 5, 6])
    c = mat3x3(1.1)
    d = mat3x3(2)
    e = double(5.0)
    f = double(8.0)

    h = vec3(5)
    idx_0 = [1]
    idx_1 = [1, 0]

    print(grad_vec(a, h))


test()