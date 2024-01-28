from common.logger import Logger
import numpy as np
import taichi as ti

from src.new_system import TiSystem
from src.common.types import *
from src.common.matrix_ops import *


@ti.data_oriented
class System2D(TiSystem):
    def solve(self):
        self.current_time = 0
        self.current_step = self.config.start_step

        if self.config.start_step == 0:
            Logger.log("Start solve initials.")

            self.initials_OT_2D()

            self.initials_ghosts()
            self.save_file(self.current_step)
            Logger.log("Initials - done!")
        else:
            Logger.log(f"Start solve from step: {self.config.start_step}.")
            self.read_file(self.config.start_step)
            Logger.log(
                f"Start solve time: {self.current_time}, end_time: {self.config.end_time}."
            )

        Logger.log("Start solving.")

        if self.div_cleaning == True:
            self.update_B_staggered_call()

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

    def get_cfl(self):
        self.fv_computer.update_data(self.rho[0], self.p[0], self.u[0], self.B[0])
        lambdas = self.fv_computer.get_cfl_cond() + vec3(1e-9)

        print(lambdas, self.h)

        dT = self.CFL * ti.min(
            self.h[0] / np.max([lambdas[0], 1]),
            self.h[1] / np.max([lambdas[1], 1]),
        )

        # return np.min([dT, self.h[0] * self.h[1]])
        return dT * self.h[0]

    @ti.kernel
    def initials_OT_2D(self):
        sq_pi = ti.sqrt(4 * ti.math.pi)
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            self.rho[0][idx] = self.RHO0
            self.u[0][idx] = vec3(
                [
                    -self.U0 * ti.math.sin(self.h[1] * y),
                    self.U0 * ti.math.sin(self.h[0] * x),
                    0,
                ]
            )
            self.B[0][idx] = (
                vec3(
                    [
                        -self.B0 * ti.math.sin(self.h[1] * y),
                        self.B0 * ti.math.sin(2.0 * self.h[0] * x),
                        0,
                    ]
                )
                / sq_pi
            )

    def FV_step(self, dT):
        self.fv_computer.update_data(self.rho[0], self.p[0], self.u[0], self.B[0])
        self.fv_computer.compute(self.rho[1], self.u[1], self.B[1], self.E)

        self.ghosts_system(self.rho[1], self.u[1], self.B[1])

        self.fv_computer.ghosts_periodic_foo_call(self.E)

        self.fv_computer.computeB_staggered(self.E, self.B_staggered[1])
        self.fv_computer.ghosts_periodic_foo_call(self.B_staggered[1])
        self.sum_fields_1_order(self.B_staggered[0], self.B_staggered[1], dT)
        # self.update_B_call(0)

        self.sum_fields_u_1_order(self.u[0], self.u[1], dT, self.rho[0])

        self.sum_fields_1_order(self.rho[0], self.rho[1], dT)
        self.sum_fields_1_order(self.B[0], self.B[1], dT)

        self.div_fields_u_1_order(self.u[0], self.rho[0])

        self.fv_computer.computeP(self.p[0], self.get_B0)
        self.fv_computer.ghosts_periodic_foo_call(self.p[0])

    def update_B_staggered_call(self, j=0):
        self.staggered_idx = j
        self.update_B_staggered_2D()
        self.fv_computer.ghosts_periodic_foo_call(self.B_staggered[self.staggered_idx])

    def update_B_call(self, j=0):
        self.staggered_idx = j
        self.update_B_2D()
        self.fv_computer.ghosts_periodic_foo_call(self.B[self.staggered_idx])

    @ti.kernel
    def update_B_staggered_2D(self):
        for idx in ti.grouped(self.B[self.staggered_idx]):
            if not self.fv_computer.check_ghost_idx(idx):
                result = vec3(0)
                for i in ti.ndrange(2):
                    idx_left = idx
                    idx_right = idx + get_basis(i)

                    result[i] = 0.5 * (
                        self.B[self.staggered_idx][idx_left][i]
                        + self.B[self.staggered_idx][idx_right][i]
                    )

                result[2] = self.B[self.staggered_idx][idx][2]
                self.B_staggered[self.staggered_idx][idx] = result

    @ti.kernel
    def update_B_2D(self):
        for idx in ti.grouped(self.B[self.staggered_idx]):
            if not self.fv_computer.check_ghost_idx(idx):
                result = vec3(0)
                for i in ti.ndrange(2):
                    idx_left = idx
                    idx_right = idx - get_basis(i)

                    result[i] = 0.5 * (
                        self.B_staggered[self.staggered_idx][idx_left][i]
                        + self.B_staggered[self.staggered_idx][idx_right][i]
                    )

                result[2] = self.B_staggered[self.staggered_idx][idx][2]
                self.B[self.staggered_idx][idx] = result
