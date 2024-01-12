import numpy as np
import taichi as ti
from src.common.boundaries import _check_ghost, get_ghost_new_idx, get_mirror_new_idx

from common.logger import Logger
from common.data_service import DataService
from src.common.pointers import get_elem_1d
from src.problem.configs import ProblemConfig, SystemConfig
from src.problem.problem import Problem

from src.common.types import *
from src.common.matrix_ops import *
from src.spatial_diff.diff_fv import div_vec_2D


@ti.data_oriented
class BaseSystem:
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
            for _ in range(self.config.rk_steps)
        ]

        self.rho = [
            ti.field(dtype=double, shape=self.config.shape)
            for _ in range(self.config.rk_steps)
        ]

        self.problem = Problem(problem_cfg)

    @ti.func
    def get_B0(self, idx):
        return self.B[0][idx]

    @ti.func
    def get_Bstag0(self, idx):
        return self.B_staggered[0][idx]

    @ti.func
    def get_Bstag1(self, idx):
        return self.B_staggered[1][idx]

    @ti.func
    def get_E(self, idx):
        return self.E[idx]

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
                self.B_staggered[0].to_numpy(),
                self.E.to_numpy(),
            ),
        )
        Logger.log(
            f"Writind step_{self.current_step}, t: {self.current_time} to file - done!"
        )

    @ti.func
    def check_ghost_idx(self, idx):
        result = False

        for axis_idx in ti.static(range(len(self.config.dim))):
            axis = self.config.dim[axis_idx]
            result = result or _check_ghost(
                get_elem_1d(self.config.shape, axis),
                get_elem_1d(self.config.ghosts, axis),
                idx[axis],
            )

        return result

    @ti.kernel
    def ghosts_periodic(self, out: ti.template(), change_sign: ti.types.int8):
        for idx in ti.grouped(out):
            if self.check_ghost_idx(idx):
                out[idx] = out[
                    get_ghost_new_idx(self.config.ghosts, self.config.shape, idx)
                ]

    @ti.kernel
    def ghosts_mirror(self, out: ti.template(), change_sign: ti.types.int8):
        for idx in ti.grouped(out):
            if self.check_ghost_idx(idx):
                res = out[
                    get_mirror_new_idx(self.config.ghosts, self.config.shape, idx)
                ]

                if change_sign != 0:
                    res *= -1.0
                out[idx] = res

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
            _u = vec3(0)
            _u[self.config.dim[0]] = -self.config.U0 * ti.math.sin(
                self.config.h[self.config.dim[1]] * idx[self.config.dim[1]]
            )
            _u[self.config.dim[1]] = self.config.U0 * ti.math.sin(
                self.config.h[self.config.dim[0]] * idx[self.config.dim[0]]
            )

            _B = vec3(0)
            _B[self.config.dim[0]] = -self.config.B0 * ti.math.sin(
                self.config.h[self.config.dim[1]] * idx[self.config.dim[1]]
            )
            _B[self.config.dim[1]] = self.config.B0 * ti.math.sin(
                2.0 * self.config.h[self.config.dim[0]] * idx[self.config.dim[0]]
            )

            self.rho[0][idx] = self.config.RHO0
            self.u[0][idx] = _u
            self.B[0][idx] = _B / sq_pi

    @ti.kernel
    def initials_SOD(self):
        sq_pi = ti.sqrt(4 * ti.math.pi)
        for idx in ti.grouped(self.rho[0]):
            rho_ = 1.0
            is_left = True

            for axis_idx in ti.static(range(len(self.config.dim))):
                axis = self.config.dim[axis_idx]
                is_left = is_left and (
                    idx[axis] < 0.5 * self.config.shape[self.config.dim[axis_idx]]
                )

            if not is_left:
                rho_ = 0.1

            self.rho[0][idx] = rho_
            self.u[0][idx] = vec3(0)
            self.B[0][idx] = vec3(0)

            self.B[0][idx][0] = 3.0 / sq_pi

            if is_left:
                self.B[0][idx][1] = 5.0 / sq_pi
            else:
                self.B[0][idx][1] = 2.0 / sq_pi

    @ti.kernel
    def convert_stag_grid(self, in_arr: ti.template(), out_arr: ti.template()):
        for idx in ti.grouped(out_arr):
            if not self.check_ghost_idx(idx):
                result = in_arr(idx)

                for axis_idx in ti.static(range(len(self.config.dim))):
                    axis = self.config.dim[axis_idx]

                    # result[axis] += in_arr(idx + get_basis(axis))[axis]
                    # result[axis] *= 0.5

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

            self.ghosts_periodic(self.rho[0], 0)
            self.ghosts_periodic(self.p[0], 0)
            self.ghosts_periodic(self.u[0], 1)
            self.ghosts_periodic(self.B[0], 0)
            self.convert_stag_grid(self.get_B0, self.B_staggered[0])

            self.save_file(self.current_step)
            Logger.log("Initials - done!")
        else:
            Logger.log(f"Start solve from step: {self.config.start_step}.")
            self.read_file(self.config.start_step)
            Logger.log(
                f"Start solve time: {self.current_time}, end_time: {self.config.end_time}."
            )

    def solve(self):
        self.current_time = 0
        self.current_step = self.config.start_step

        self.initials()

        Logger.log("Start solving.")

        self.ghosts_periodic(self.B[0], 0)
        self.convert_stag_grid(self.get_B0, self.B_staggered[0])

        Logger.log("ghosts_periodic.")
        while self.current_time < self.config.end_time or (
            self.current_step % self.config.rw_del != 0
        ):
            dT = self.get_cfl()
            print(f"CFL dT: {dT}")

            self.current_time += dT
            self.current_step += 1

            self.FV_step(dT)

            mean_div = np.mean(
                self.p[0].to_numpy()[
                    self.config.ghosts[0] : -self.config.ghosts[0],
                    self.config.ghosts[1] : -self.config.ghosts[2],
                    :,
                ]
            )
            print(f"Mean divB: {mean_div}")

            if self.current_step % self.config.rw_del == 0:
                self.save_file(self.current_step)
