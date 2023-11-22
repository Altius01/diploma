import taichi as ti

from taichi_src.common.types import *
from taichi_src.common.matrix_ops import static_get_len, static_get_shape

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


# @ti.func
# def get_elem(arr: ti.template(), idx) -> double:
#     idx_len = ti.static(static_get_len(idx))
    
#     if ti.static(idx_len==1):
#         return get_elem_1d(arr, idx)
#     elif ti.static(idx_len==2):
#         return get_elem_2d(arr, idx)
#     elif ti.static(idx_len==3):
#         return get_elem_3d(arr, idx)