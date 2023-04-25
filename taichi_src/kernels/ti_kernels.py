import taichi as ti

from taichi_src.kernels.common.common_ti_kernels import * 
from taichi_src.kernels.filters.filter_kernels import *

@ti.kernel
def rho_u(rho: ti.types.ndarray(), u: ti.types.ndarray(), result: ti.types.ndarray()):
    for i in ti.grouped(result):
        result[i] = rho[get_sc_idx(i)]*u[i]

@ti.kernel
def BiBj(B: ti.types.ndarray(), result: ti.types.ndarray()):
    for idx in ti.grouped(result):
        i, j, x, y, z = idx
        result[idx] = B[i, x, y, z]*B[j, x, y, z]

@ti.kernel
def abs_S_compute(S: ti.types.ndarray(), result: ti.types.ndarray()):
    for idx in ti.grouped(result):
        x, y, z = idx
        for i, j in ti.ndrange(3, 3):
            result[idx] += S[i, j, x, y, z]*S[i, j, x, y, z]

    for idx in ti.grouped(result):
        result[idx] = ti.sqrt(2 * result[idx])
