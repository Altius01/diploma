import numpy as np
import taichi as ti

from taichi_src.common.types import *
from taichi_src.common.matrix_ops import *

@ti.kernel
def knl_box_filter_1D(arr: ti.template(), output: ti.template(), 
    is_ghost_foo: ti.template(), eps: double, axe: int):
    ti.static_assert(arr.shape == output.shape)

    for idx in arr:
        if not is_ghost_foo(idx):
            left = idx + get_basis(axe)
            right = idx - get_basis(axe)

            out[idx] = (0.5*(arr(left) + arr(right))*eps**2 + (12 - eps**2)*arr(idx)) / 12

@ti.kernel
def favre_filter_divide(arr: ti.template(), rho_filtered: ti.template()):
    for idx in arr:
        arr[idx] = arr[idx] / rho_filtered[idx]

def box_filter_1D(arr, out, is_ghost_foo, eps):
    knl_box_filter_1D(arr, out, is_ghost_foo, eps, 0)

def box_filter_2D(arr, out, is_ghost_foo, eps):
    for i in range(2):
        knl_box_filter_1D(arr, out, is_ghost_foo, eps, i)

def box_filter_3D(arr, out, is_ghost_foo, eps):
    for i in range(3):
        knl_box_filter_1D(arr, out, is_ghost_foo, eps, i)
