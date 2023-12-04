from abc import ABC, abstractmethod
import re

import taichi as ti
from src.common.matrix_ops import get_idx_to_basis, get_mat_col, get_vec_col

from src.common.types import *


@ti.func
def parse_vars(v: vec7):
    return v[0], vec3(v[1:4]), vec3(v[4:])


@ti.data_oriented
class Flux(ABC):
    def __init__(self, h, filter_size=vec3i(1), les=NonHallLES.DNS):
        self.h = h
        self.filter_size = filter_size
        self.filter_delta = self.h * self.filter_size
        self.les = les

    @ti.func
    @abstractmethod
    def flux_convective(self, q):
        ...

    @ti.func
    @abstractmethod
    def flux_viscous(self, v, grad_U, grad_B):
        ...

    @ti.func
    @abstractmethod
    def flux_hall(self, v, grad_B, rot_B):
        ...

    @ti.func
    @abstractmethod
    def flux_les(self, v, grad_U, grad_B, rot_U, rot_B):
        ...


@ti.data_oriented
class RhoFlux(Flux):
    @ti.func
    def flux_convective(self, q):
        result = vec3(q[1:4])
        return result

    @ti.func
    def flux_viscous(self, v, grad_U, grad_B):
        return vec3(0)

    @ti.func
    def flux_hall(self, v, grad_B, rot_B):
        return vec3(0)

    @ti.func
    def flux_les(self, v, grad_U, grad_B, rot_U, rot_B):
        return vec3(0)


@ti.data_oriented
class MomentumFlux(Flux):
    def __init__(self, Re, Ma, Ms, gamma, h, filter_size=vec3i(1), les=NonHallLES.DNS):
        super().__init__(h, filter_size=filter_size, les=les)
        self.Re = Re
        self.Ma = Ma
        self.Ms = Ms
        self.gamma = gamma

    @ti.func
    def get_sound_speed(self, p, rho):
        return (1.0 / self.Ms) * ti.sqrt(self.gamma * p / rho)

    @ti.func
    def get_pressure(self, rho):
        return ti.pow(rho, self.gamma)

    @ti.func
    def flux_convective(self, q):
        q_rho, q_u, q_b = parse_vars(q)

        p = self.get_pressure(q_rho)
        BB = q_b.outer_product(q_b)
        rho_UU = q_u.outer_product(q_u) / q_rho
        result = rho_UU + (p + (0.5 / self.Ma**2) * q_b.norm_sqr()) * kron - BB
        return result

    @ti.func
    def flux_viscous(self, v, grad_U, grad_B):
        divU = grad_U.trace()

        return (grad_U + grad_U.transpose() + (2.0 / 3.0) * divU * kron) / self.Re

    @ti.func
    def flux_hall(self, v, grad_B, rot_B):
        return mat3x3(0)

    @ti.func
    def flux_les(self, v, grad_U, grad_B, rot_U, rot_B):
        return mat3x3(0)


@ti.data_oriented
class MagneticFlux(Flux):
    def __init__(self, Rem, delta_hall, h, filter_size=vec3i(1), les=NonHallLES.DNS):
        super().__init__(h, filter_size=filter_size, les=les)
        self.Rem = Rem
        self.delta_hall = delta_hall

    @ti.func
    def flux_convective(self, q):
        q_rho, q_u, q_b = parse_vars(q)

        Bu = q_b.outer_product(q_u) / q_rho
        result = Bu - Bu.transpose()
        return result

    @ti.func
    def flux_viscous(self, v, grad_U, grad_B):
        return (grad_B - grad_B.transpose()) / self.Rem

    @ti.func
    def flux_hall(self, v, grad_B, rot_B):
        v_rho, v_u, v_b = parse_vars(v)

        j = rot_B
        v_h = -self.delta_hall * j / v_rho
        v_hB = v_h.outer_product(v_b)
        return v_hB - v_hB.transpose()

    @ti.func
    def flux_les(self, v, grad_U, grad_B, rot_U, rot_B):
        return mat3x3(0)
