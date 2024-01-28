from typing import List

import taichi as ti
from src.common.matrix_ops import get_mat_col
from src.problem.configs import ProblemConfig
from src.reconstruction.reconstructors import (
    FirstOrder,
    Reconstructor,
    SecondOrder,
    Weno5,
)

from src.common.types import *
from src.flux.flux import MagneticFlux, MomentumFlux, RhoFlux
from src.flux.solvers import HLLDSolver, RoeSolver, Solver
from src.spatial_diff.diff_fv import (
    V_plus_sc_2D,
    V_plus_vec7_2D,
    V_plus_vec_2D,
    get_dx_st_2D,
    grad_vec_2D,
    rot_vec_2D,
)


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
            Weno5(axis)
            # for axis in self.cfg.dim
            # FirstOrder(axis)
            for axis in self.cfg.dim
        ]

        self.solvers: List[Solver] = [
            HLLDSolver(self.rho_computer, self.u_computer, self.B_computer, axis)
            # RoeSolver(self.rho_computer, self.u_computer, self.B_computer, axis)
            for axis in self.cfg.dim
        ]

        self.get_flux_right: List[ti.func] = [
            self.get_flux_right(axis) for axis in self.cfg.dim
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

    # @ti.func
    # def v_rho(self, idx):
    #     return self.rho[idx]

    @ti.func
    def v_u(self, idx):
        return self.u[idx]

    @ti.func
    def v_b(self, idx):
        return self.B[idx]

    @ti.func
    def grad_U(self, idx):
        return grad_vec_2D(self.v_u, self.cfg.h, idx)

    @ti.func
    def grad_B(self, idx):
        return grad_vec_2D(self.v_b, self.cfg.h, idx)

    @ti.func
    def rot_U(self, idx):
        return rot_vec_2D(self.v_u, self.cfg.h, idx)

    @ti.func
    def rot_B(self, idx):
        return rot_vec_2D(self.v_b, self.cfg.h, idx)

    @ti.func
    def get_s_j_max(self, idx):
        result = vec3(0)

        for axis_idx in ti.static(range(len(self.cfg.dim))):
            axis = self.cfg.dim[axis_idx]

            result[axis] = self.solvers[axis_idx].get_max_eigenval(self.q(idx))
            u = self.u[idx]
            result[axis] += ti.abs(u[axis])

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

    def get_flux_right(self, axis_idx):
        @ti.func
        def _get_flux_right(idx_l, idx_r):
            result = self.solvers[axis_idx].get_conv(
                q_l=self.q(idx_l), q_r=self.q(idx_r)
            )

            # if ti.static(len(self.solvers) > 1):
            #     if axis_idx == 1:
            #         result = self.solvers[1].get_conv(
            #             q_l=self.q(idx_l), q_r=self.q(idx_r)
            #         )

            # if ti.static(len(self.solvers) > 2):
            #     if axis_idx == 2:
            #         result = self.solvers[2].get_conv(
            #             q_l=self.q(idx_l), q_r=self.q(idx_r)
            #         )

            corner = idx_l
            v = self.v(corner)

            gradU = self.grad_U(corner)
            gradB = self.grad_B(corner)

            result[1:4] -= get_mat_col(
                self.u_computer.flux_viscous(v, gradU, gradB),
                axis_idx,
            )

            result[4:] -= get_mat_col(
                self.B_computer.flux_viscous(v, gradU, gradB),
                axis_idx,
            )

            return result

        return _get_flux_right
