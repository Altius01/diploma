from dataclasses import dataclass
from typing import List
import numpy as np
import taichi as ti
from common.boundaries import _check_ghost, get_ghost_new_idx

from common.config import Config

from common.logger import Logger
from common.data_service import DataService
from reconstruction.reconstructors import Reconstructor
from src.problem.new_problem import Problem
from src.problem.problem import ProblemConfig

from src.common.types import *
from src.common.matrix_ops import *


@dataclass
class SystemConfig:
    CFL: float

    rk_steps: int

    initials: str
    les_model: str

    hall: bool
    ideal: bool

    h: List[float]

    shape: List[int]
    ghosts: List[int]
    domain_size: List[int]

    RHO0: float
    U0: float
    B0: float
    eps_p: float

    gamma: float
    Re: float
    Ms: float
    Ma: float
    Rem: float
    nu_0: float
    delta_hall: float

    rw_del: int
    end_time: float
    start_step: int

    def __init__(self, config: Config):
        self.RHO0 = config.RHO0
        self.U0 = config.U0
        self.B0 = config.B0
        self.eps_p = 1e-5
        self.h = [0, 0, 0]
        self.Re = config.Re
        self.nu_0 = config.nu_0
        self.Rem = config.Rem
        self.delta_hall = config.delta_hall
        self.Ma = config.Ma
        self.Ms = config.Ms
        self.gamma = config.gamma

        self.CFL = config.CFL

        self.rw_del = config.rw_del

        self.end_time = config.end_time
        self.start_step = config.start_step

        self.rk_steps = 2
        self.les_model = NonHallLES(config.model)
        self.initials = Initials(config.initials)
        self.ideal = config.ideal
        self.hall = config.hall

        self.dim = len(self.shape)

        self.debug_fv_step = True

        self.config = config
        self.ghost = config.ghosts
        self.shape = config.shape
        self.true_shape = config.true_shape


@ti.data_oriented
class System:
    def __init__(self, sys_cfg: SystemConfig, data_path="", arch=ti.cpu):
        self.config: SystemConfig = sys_cfg
        problem_cfg = ProblemConfig(sys_cfg)

        for i, l in enumerate(self.config.domain_size):
            self.config.h[i] = l / self.config.true_shape[i]

        self.data_service = DataService(dir_name=data_path)

        self.u = [
            ti.Vector.field(n=3, dtype=double, shape=self.config.shape)
            for _ in range(self.config.rk_steps)
        ]

        self.B = [
            ti.Vector.field(n=3, dtype=double, shape=self.config.shape)
            for _ in range(self.config.rk_steps)
        ]

        self.B_staggered = [
            ti.Vector.field(n=3, dtype=double, shape=self.config.shape)
            for _ in range(self.config.rk_steps)
        ]

        self.E = ti.Vector.field(n=3, dtype=double, shape=self.config.shape)

        self.p = [
            ti.field(dtype=double, shape=self.config.shape)
            for _ in range(self.rk_steps)
        ]

        self.rho = [
            ti.field(dtype=double, shape=self.config.shape)
            for _ in range(self.rk_steps)
        ]

        self.problem = Problem(problem_cfg)

    def read_file(self, i):
        (
            self.current_time,
            rho_,
            p_,
            u_,
            B_,
        ) = self.data_service.read_data(i)

        self.rho[0].from_numpy(rho_)
        self.p[0].from_numpy(p_)

        self.u[0].from_numpy(u_)
        self.B[0].from_numpy(B_)

    def save_file(self, i):
        Logger.log(f"Writing step_{self.current_step}, t: {self.current_time} to file.")
        self.data_service.save_data(
            i,
            (
                self.current_time,
                self.rho[0].to_numpy(),
                self.p[0].to_numpy(),
                self.u[0].to_numpy(),
                self.B[0].to_numpy(),
            ),
        )
        Logger.log(
            f"Writind step_{self.current_step}, t: {self.current_time} to file - done!"
        )

    @ti.func
    def check_ghost_idx(self, shape, idx):
        result = False

        for i in ti.ndrange(self.config.dim):
            result = result or _check_ghost(shape[i], idx[i])

        return result

    @ti.kernel
    def ghosts_periodic(self, out: ti.template()):
        for idx in ti.grouped(out):
            if self.check_ghost_idx(idx):
                out[idx] = out[
                    get_ghost_new_idx(self.config.ghosts, self.config.shape, idx)
                ]

    @ti.kernel
    def sum_fields_1_order(self, a: ti.template(), b: ti.template(), c1: double):
        for idx in ti.grouped(a):
            a[idx] = a[idx] + c1 * b[idx]

    @ti.kernel
    def sum_fields_u_1_order(
        self, a: ti.template(), b: ti.template(), c1: double, rho_old: ti.template()
    ):
        for idx in ti.grouped(a):
            a[idx] = a[idx] * rho_old[idx] + c1 * b[idx]

    @ti.kernel
    def div_fields_u_1_order(self, a: ti.template(), rho_new: ti.template()):
        for idx in ti.grouped(a):
            a[idx] /= rho_new[idx]

    @ti.kernel
    def initials_OT(self):
        sq_pi = ti.sqrt(4 * ti.math.pi)
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            self.rho[0][idx] = self.config.RHO0
            self.u[0][idx] = vec3(
                [
                    -self.config.U0 * ti.math.sin(self.config.h[1] * y),
                    self.config.U0 * ti.math.sin(self.confi[0] * x),
                    0,
                ]
            )
            self.B[0][idx] = (
                vec3(
                    [
                        -self.config.B0 * ti.math.sin(self.config.h[1] * y),
                        self.config.B0 * ti.math.sin(2.0 * self.config.h[0] * x),
                        0,
                    ]
                )
                / sq_pi
            )

    @ti.kernel
    def convert_stag_grid(self, in_arr: ti.template(), out_arr: ti.template()):
        for idx in ti.grouped(out_arr):
            if not self.check_ghost_idx(idx):
                result = in_arr[idx]

                for i in ti.ndrange(self.config.dim):
                    reco: Reconstructor = self.problem.reconstructors[i]

                    result[i] = 0.5 * (
                        reco.get_right(idx) + reco.get_left(idx + get_basis(i))
                    )

                out_arr[idx] = result

    def get_cfl(self):
        self.problem.update_data(self.rho[0], self.p[0], self.u[0], self.B[0])
        lambdas = self.problem.get_cfl_cond()

        dT = self.config.CFL * np.min(
            [self.config.h[i] / lambdas[i] for i in self.config.dim]
        )

        return dT

    def initials(self):
        if self.config.start_step == 0:
            Logger.log("Start solve initials.")

            self.initials_OT()

            self.ghosts_periodic(self.rho[0])
            self.ghosts_periodic(self.p[0])
            self.ghosts_periodic(self.u[0])
            self.ghosts_periodic(self.B[0])

            self.save_file(self.current_step)
            Logger.log("Initials - done!")
        else:
            Logger.log(f"Start solve from step: {self.config.start_step}.")
            self.read_file(self.config.start_step)
            Logger.log(
                f"Start solve time: {self.current_time}, end_time: {self.config.end_time}."
            )

    @ti.kernel
    def computeP(self, out: ti.template(), foo_B: ti.template()):
        for idx in ti.grouped(out):
            if not self.check_ghost_idx(idx):
                out[idx] = self.div_vec(foo_B, self.config.h, idx)

    @ti.kernel
    def compute(
        self,
        out_rho: ti.template(),
        out_u: ti.template(),
        out_B: ti.template(),
        out_E: ti.template(),
    ):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                flux = vec7(0)
                for axis in range(self.config.dim):
                    flux_r = self.problem.get_flux_right(idx, idx + get_basis(axis))
                    flux_l = self.problem.get_flux_right(idx - get_basis(axis), idx)

                    flux += (flux_l - flux_r) / self.config.h[axis]

    @ti.kernel
    def computeB_staggered(self, E: ti.template(), B_stag_out: ti.template()):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                i, j, k = idx

                ijk = vec3i([i, j, k])
                ijm1k = vec3i([i, j - 1, k])
                im1jm1k = vec3i([i - 1, j - 1, k])

                res = vec3(0)
                res[0] = -((E[ijk][2] - E[ijm1k][2])) / (self.h[1])

                res[1] = -((E[im1jm1k][2] - E[ijm1k][2])) / (self.h[0])

                B_stag_out[idx] = res

    def FV_step(self, dT):
        self.problem.update_data(self.rho[0], self.p[0], self.u[0], self.B[0])

        self.compute(self.rho[1], self.u[1], self.B[1], self.E)

        self.computeB_staggered(self.E, self.B_staggered[1])
        self.sum_fields_1_order(self.B_staggered[0], self.B_staggered[1], dT)
        self.ghosts_periodic(self.B_staggered[0])

        self.sum_fields_u_1_order(self.u[0], self.u[1], dT, self.rho[0])

        self.sum_fields_1_order(self.rho[0], self.rho[1], dT)

        self.div_fields_u_1_order(self.u[0], self.rho[0])

        self.convert_stag_grid(in_arr=self.B_staggered[0], out_arr=self.B[0])
        self.ghosts_periodic(self.B_staggered[0])

        self.computeP(self.p[0], ti.func(lambda idx: self.B[0][idx]))
        self.ghosts_periodic(self.p[0])

    def solve(self):
        self.current_time = 0
        self.current_step = self.config.start_step

        self.initials()

        Logger.log("Start solving.")

        self.convert_stag_grid(in_arr=self.B[0], out_arr=self.B_staggered[0])
        self.ghosts_periodic(self.B_staggered[0])

        while self.current_time < self.config.end_time or (
            self.current_step % self.config.rw_del != 0
        ):
            dT = self.get_cfl()
            # if self.debug_fv_step:
            #     print(f"CFL: dT: {dT}")

            self.current_time += dT
            self.current_step += 1

            self.FV_step(dT)

            if self.current_step % self.config.rw_del == 0:
                self.save_file(self.current_step)
