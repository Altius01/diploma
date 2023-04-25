import taichi as ti

from taichi_src.kernels.ti_kernels import *
from taichi_src.kernels.common.common_ti_kernels import ti_kron, get_sc_idx

@ti.kernel
def Lu_A(rho: ti.types.ndarray(), rho_u: ti.types.ndarray(), result: ti.types.ndarray()):
    for idx in ti.grouped(result):
        i, j, x, y, z = idx
        result[idx] = (rho_u[i, x, y, z] * rho_u[j, x, y, z]) / rho[get_sc_idx(idx)]

@ti.kernel
def Lb_A(rho: ti.types.ndarray(), rho_u: ti.types.ndarray(), B: ti.types.ndarray(), result: ti.types.ndarray()):
    for idx in ti.grouped(result):
        i, j, x, y, z = idx
        result[idx] = (rho_u[i, x, y, z] * B[j, x, y, z]) / rho[get_sc_idx(idx)]

@ti.kernel
def Lu_compute(Ma:ti.f64,
                Lu: ti.types.ndarray(),
                BiBj_: ti.types.ndarray(),
                BiBj_filter: ti.types.ndarray(),
                Lu_A_: ti.types.ndarray(),
                Lu_A_filter: ti.types.ndarray(),):
    for idx in ti.grouped(Lu):
        Lu[idx] = (Lu_A_filter[idx] - Lu_A_[idx]) - (BiBj_filter[idx] - BiBj_[idx])/Ma**2

@ti.kernel
def Lb_compute(Lb: ti.types.ndarray(),
                A: ti.types.ndarray(),
                A_filtered: ti.types.ndarray(),):
    for idx in ti.grouped(Lb):
        i, j, x, y, z = idx
        Lb[idx] = (A[idx] - A_filtered[idx]) - (A[j, i, x, y, z] - A_filtered[j, i, x, y, z])

@ti.kernel
def M_A_compute(M_A: ti.types.ndarray(),
                alpha: ti.types.ndarray(),
                S: ti.types.ndarray(),):
    for idx in ti.grouped(M_A):
        i, j, x, y, z = idx
        M_A[idx] = alpha[idx] * S[idx] * (1 - ti_kron(i, j) /3.0)

@ti.kernel
def m_A_compute(m_A: ti.types.ndarray(),
                phi: ti.types.ndarray(),
                J: ti.types.ndarray(),):
    for idx in ti.grouped(m_A):
        m_A[idx] = phi[idx] * J[idx]

@ti.kernel
def M_compute(M: ti.types.ndarray(),
                A: ti.types.ndarray(),
                A_filtered: ti.types.ndarray(),):
    for idx in ti.grouped(M):
        i, j, x, y, z = idx
        M[idx] = A[idx] - A_filtered[idx]
