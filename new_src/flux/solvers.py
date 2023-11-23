from abc import ABC, abstractmethod

import taichi as ti
from new_src.common.matrix_ops import get_mat_col, get_vec_col
from new_src.flux.flux import MagneticFlux, MomentumFlux, RhoFlux

from new_src.common.types import *


@ti.data_oriented
class Solver(ABC):
    def __init__(self, flux_rho: RhoFlux, flux_u: MomentumFlux, flux_b: MagneticFlux):
        self.flux_rho: RhoFlux = flux_rho
        self.flux_u: MomentumFlux = flux_u
        self.flux_b: MagneticFlux = flux_b

    @ti.func
    @abstractmethod
    def get_conv(self, q_l: vec7, q_r: vec7):
        raise NotImplementedError

    @ti.func
    def get_viscous(self, v, grad_U, grad_B):
        viscous_flux = vec7(0)
        viscous_flux[1:4] = self.flux_u.flux_viscous(v, grad_U, grad_B)
        viscous_flux[4:] = self.flux_b.flux_viscous(v, grad_U, grad_B)

        return viscous_flux

    @ti.func
    def get_hall(self, v, grad_B, rot_B):
        hall_flux = vec7(0)
        hall_flux[4:] = self.flux_b.flux_hall(v, grad_B, rot_B)

        return hall_flux

    @ti.func
    def get_les(self, v, grad_U, grad_B, rot_U, rot_B):
        les_flux = vec7(0)
        les_flux[1:4] = self.flux_u.flux_les(v, grad_U, grad_B, rot_U, rot_B)
        les_flux[4:] = self.flux_b.flux_les(v, grad_U, grad_B, rot_U, rot_B)

        return les_flux


@ti.data_oriented
class RoeSolver(Solver):
    @ti.func
    def get_conv(self, q_l: vec7, q_r: vec7):
        conv_flux = vec7(0)

        s_max = ti.max(
            self.flux_u.get_c_fast(q_l, ti.static(self.axes)),
            self.flux_u.get_c_fast(q_r, ti.static(self.axes)),
        )

        conv_flux[0] = get_vec_col(
            self.flux_rho.flux_convective(q_l) + self.flux_rho.flux_convective(q_r),
            self.axes,
        )

        conv_flux[1:4] = get_mat_col(
            self.flux_u.flux_convective(q_l) + self.flux_u.flux_convective(q_r),
            self.axes,
        )

        conv_flux[4:] = get_mat_col(
            self.flux_b.flux_convective(q_l) + self.flux_b.flux_convective(q_r),
            self.axes,
        )

        conv_flux -= s_max * (q_r - q_l)
        return 0.5 * conv_flux
