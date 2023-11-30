from dataclasses import dataclass
import taichi as ti
from problem.new_system import SystemConfig
from src.common.boundaries import get_ghost_new_idx, get_mirror_new_idx
from src.common.matrix_ops import get_basis, get_mat_col
from src.common.pointers import get_elem_1d
from src.common.types import *
from src.flux.flux import MagneticFlux, MomentumFlux, RhoFlux
from src.flux.solvers import RoeSolver, Solver
from src.reconstruction.reconstructors import tvd_1order, tvd_slope_limiter_2order
from src.spatial_diff.diff_fv import (
    V_plus_sc_2D,
    V_plus_vec_2D,
    get_dx_st_2D,
    grad_sc_2D,
    grad_vec_2D,
    rot_vec_2D,
)


@dataclass
class ProblemConfig:
    gamma: double
    Re: double
    Ms: double
    Ma: double
    Rem: double
    delta_hall: double

    ghosts: int

    shape: tuple
    h: tuple
    domain_size: tuple

    ideal: bool
    hall: bool

    dim: int

    def __init__(self, sys_cfg: SystemConfig):
        self.gamma = sys_cfg.gamma
        self.Re = sys_cfg.Re
        self.Ms = sys_cfg.Ms
        self.Ma = sys_cfg.Ma
        self.Rem = sys_cfg.Rem
        self.delta_hall = sys_cfg.delta_hall
        self.ghost = sys_cfg.ghost
        self.config = sys_cfg.config
        self.h = sys_cfg.h
        self.config = sys_cfg.config
        self.ideal = sys_cfg.ideal
        self.hall = sys_cfg.hall
        self.dim = sys_cfg.dim


@ti.data_oriented
class Problem:
    def __init__(self, cfg: ProblemConfig) -> None:
        self.cfg = cfg

        self.h = vec3(cfg.h)
        self.shape = cfg.shape
        self.ghost = cfg.ghosts
        self.domain_size = cfg.domain_size

        self.Re = cfg.Re
        self.Ms = cfg.Ms

        self.Ma = cfg.Ma
        self.Rem = cfg.Rem

        self.delta_hall = cfg.delta_hall

        self.gamma = cfg.gamma

        self.ideal = cfg.ideal
        self.hall = cfg.hall
        self.les = cfg.les

        self.filter_size = vec3i([1, 1, 1])
        # self.k = -(1.0/3.0)
        self.k = -1.0

        self.dimensions = cfg.dim

        self.rho_computer = RhoFlux(self.h, filter_size=self.filter_size, les=self.les)
        self.u_computer = MomentumFlux(
            self.Re,
            self.Ma,
            self.Ms,
            self.gamma,
            self.h,
            filter_size=self.filter_size,
            les=self.les,
        )
        self.B_computer = MagneticFlux(
            self.Rem,
            self.delta_hall,
            self.h,
            filter_size=self.filter_size,
            les=self.les,
        )

        self.solvers: list[Solver] = [
            RoeSolver(self.rho_computer, self.u_computer, self.B_computer, axes)
            for axes in range(self.dimensions)
        ]

    def update_data(self, rho, p, u, B):
        self.u = u
        # self.p = p
        self.B = B
        self.rho = rho

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
    def grad_rho(self, idx):
        return grad_sc_2D(self.v_rho, self.h, idx)

    @ti.func
    def grad_U(self, idx):
        return grad_vec_2D(self.v_u, self.h, idx)

    @ti.func
    def grad_B(self, idx):
        return grad_vec_2D(self.v_b, self.h, idx)

    @ti.func
    def rot_U(self, idx):
        return rot_vec_2D(self.v_u, self.h, idx)

    @ti.func
    def rot_B(self, idx):
        return rot_vec_2D(self.v_b, self.h, idx)

    @ti.func
    def _check_ghost(self, shape, ghost, idx):
        return (idx < ghost) or (idx >= shape - ghost)

    @ti.func
    def check_ghost_idx(self, idx):
        result = False

        result = result or self._check_ghost(self.shape[0], self.ghost[0], idx[0])
        if self.dimensions > 1:
            result = result or self._check_ghost(self.shape[1], self.ghost[1], idx[1])
        if self.dimensions > 2:
            result = result or self._check_ghost(self.shape[2], self.ghost[2], idx[2])

        return result

    @ti.func
    def get_s_j_max(self, idx):
        result = vec3(0)

        for axes in ti.static(range(self.dimensions)):
            result[axes] = ti.abs(self.u[idx][ti.static(axes)]) + self.solvers[
                axes
            ].get_c_fast(self.q(idx))

        return result
