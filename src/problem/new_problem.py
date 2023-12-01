from typing import List

import taichi as ti
from src.problem.configs import ProblemConfig
from src.reconstruction.reconstructors import FirstOrder, Reconstructor

from src.common.types import *
from src.flux.flux import MagneticFlux, MomentumFlux, RhoFlux
from src.flux.solvers import RoeSolver, Solver


@ti.data_oriented
class Problem:
    def __init__(self, cfg: ProblemConfig) -> None:
        self.cfg = cfg

        self.rho_computer = RhoFlux(
            cfg.h,
            filter_size=cfg.filter_size,
            les=cfg.les,
        )
        self.u_computer = MomentumFlux(
            cfg.Re,
            cfg.Ma,
            cfg.Ms,
            cfg.gamma,
            cfg.h,
            filter_size=cfg.filter_size,
            les=cfg.les,
        )
        self.B_computer = MagneticFlux(
            cfg.Rem,
            cfg.delta_hall,
            cfg.h,
            filter_size=cfg.filter_size,
            les=cfg.les,
        )

        self.reconstructors: List[Reconstructor] = [
            FirstOrder(axis) for axis in range(self.cfg.dim)
        ]

        self.solvers: List[Solver] = [
            RoeSolver(self.rho_computer, self.u_computer, self.B_computer, axis)
            for axis in range(self.cfg.dim)
        ]

    def update_data(self, rho, p, u, B):
        self.u: ti.MatrixField = u
        self.B: ti.MatrixField = B

        self.p: ti.ScalarField = p
        self.rho: ti.ScalarField = rho

    @ti.func
    def q(self, idx):
        result = vec7(0)
        result[0] = self.rho[idx]
        result[1:4] = result[0] * self.u[idx]
        result[4:] = self.B[idx]
        return result

    @ti.func
    def v(self, idx):
        result = vec7(0)
        result[0] = self.rho[idx]
        result[1:4] = self.u[idx]
        result[4:] = self.B[idx]
        return result

    @ti.func
    def q_rho(self, idx):
        return self.rho[idx]

    @ti.func
    def q_u(self, idx):
        return self.rho[idx] * self.u[idx]

    @ti.func
    def q_b(self, idx):
        return self.B[idx]

    @ti.func
    def v_rho(self, idx):
        return self.rho[idx]

    @ti.func
    def v_u(self, idx):
        return self.u[idx]

    @ti.func
    def v_b(self, idx):
        return self.B[idx]

    @ti.func
    def get_s_j_max(self, idx):
        result = vec3(0)

        for axes in ti.static(range(self.cfg.dim)):
            result[axes] = ti.abs(self.u[idx][axes]) + self.solvers[
                axes
            ].get_max_eigenval(self.q(idx))

        return result

    @ti.kernel
    def get_cfl_cond(self) -> vec3:
        res_x = double(0.0)
        res_y = double(0.0)
        res_z = double(0.0)

        for idx in ti.grouped(self.rho):
            s_j_max = self.get_s_j_max(idx)
            ti.atomic_max(res_x, s_j_max[0])
            ti.atomic_max(res_y, s_j_max[1])
            ti.atomic_max(res_z, s_j_max[2])

        return vec3([res_x, res_y, res_z])

    @ti.func
    def get_flux_right(self, idx_l, idx_r, axis):
        result = self.solvers[0].get_conv(q_l=self.q(idx_l), q_r=self.q(idx_r))
        if ti.static(self.cfg.dim > 1):
            if axis == 1:
                result = self.solvers[1].get_conv(q_l=self.q(idx_l), q_r=self.q(idx_r))

        if ti.static(self.cfg.dim > 2):
            if axis == 2:
                result = self.solvers[2].get_conv(q_l=self.q(idx_l), q_r=self.q(idx_r))

        return result
