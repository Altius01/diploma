from dataclasses import dataclass
import taichi as ti
from new_src.common.types import *
from new_src.flux.flux import MagneticFlux, MomentumFlux, RhoFlux


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
    les: NonHallLES

    dim: int


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
        return self.grad_sc(self.v_rho, self.h, idx)

    @ti.func
    def grad_U(self, idx):
        return self.grad_vec(self.v_u, self.h, idx)

    @ti.func
    def grad_B(self, idx):
        return self.grad_vec(self.v_b, self.h, idx)

    @ti.func
    def rot_U(self, idx):
        return self.rot_vec(self.v_u, self.h, idx)

    @ti.func
    def rot_B(self, idx):
        return self.rot_vec(self.v_b, self.h, idx)

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
