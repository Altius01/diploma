import numpy as np
import taichi as ti
from src.common.boundaries import _check_ghost, get_ghost_new_idx, get_mirror_new_idx

from common.logger import Logger
from common.data_service import DataService
from src.common.pointers import get_elem_1d
from src.problem.base_system import BaseSystem
from src.problem.configs import ProblemConfig, SystemConfig
from src.problem.problem import Problem

from src.common.types import *
from src.common.matrix_ops import *
from src.spatial_diff.diff_fv import div_vec_2D


@ti.data_oriented
class System(BaseSystem):
    @ti.kernel
    def computeP(self, out: ti.template(), foo_B: ti.template()):
        for idx in ti.grouped(out):
            if not self.check_ghost_idx(idx):
                out[idx] = div_vec_2D(foo_B, self.config.h, idx)

    @ti.kernel
    def compute(
        self,
        out_rho: ti.template(),
        out_u: ti.template(),
        out_B: ti.template(),
        out_E: ti.template(),
    ):
        ti_dim = ti.Vector(self.config.dim)
        for idx in ti.grouped(self.rho[0]):
            if not self.check_ghost_idx(idx):
                emf = vec3(0)
                emf_flux_l = mat3x3(0)
                emf_flux_r = mat3x3(0)

                flux = vec7(0)

                for axis_idx in ti.static(range(ti_dim.n)):
                    axis = ti_dim[axis_idx]

                    flux_r = self.problem.get_flux_right[axis_idx](
                        idx, idx + get_basis(axis)
                    )
                    flux_l = self.problem.get_flux_right[axis_idx](
                        idx - get_basis(axis), idx
                    )

                    flux += (flux_l - flux_r) / get_elem_1d(self.config.h, axis)

                    emf_flux_l[axis, :] = flux_r[4:]

                    for shift_axis_idx in range(ti_dim.n):
                        shift_axis = ti_dim[shift_axis_idx]

                        flux_r_ = self.problem.get_flux_right[axis_idx](
                            idx + get_basis(shift_axis),
                            idx + get_basis(axis) + get_basis(shift_axis),
                        )
                        emf_flux_r[axis, shift_axis] = flux_r_[4 + shift_axis]

                emf[0] = (
                    emf_flux_l[2, 1]
                    + emf_flux_r[2, 1]
                    - emf_flux_l[1, 2]
                    - emf_flux_r[1, 2]
                )
                emf[1] = (
                    emf_flux_l[0, 2]
                    + emf_flux_r[0, 2]
                    - emf_flux_l[2, 0]
                    - emf_flux_r[2, 0]
                )
                emf[2] = (
                    emf_flux_l[1, 0]
                    + emf_flux_r[1, 0]
                    - emf_flux_l[0, 1]
                    - emf_flux_r[0, 1]
                )

                out_E[idx] = 0.25 * emf

                out_rho[idx] = flux[0]
                out_u[idx] = flux[1:4]

                out_B[idx] = flux[4:]

    @ti.kernel
    def computeB_staggered(self, E: ti.template(), B_stag_out: ti.template()):
        for idx in ti.grouped(self.rho[0]):
            if not self.check_ghost_idx(idx):
                i, j, k = idx

                ijk = vec3i([i, j, k])
                ijp1k = vec3i([i, j + 1, k])
                ijm1k = vec3i([i, j - 1, k])
                ip1jk = vec3i([i + 1, j, k])
                im1jk = vec3i([i - 1, j, k])

                res = vec3(0)

                res[0] = -((E[ijk][2] - E[ijm1k][2])) / ((self.config.h[1]))

                res[1] = +((E[ijk][2] - E[im1jk][2])) / ((self.config.h[0]))

                B_stag_out[idx] = res

    def FV_step(self, dT):
        self.problem.update_data(self.rho[0], self.p[0], self.u[0], self.B[0])

        self.compute(self.rho[1], self.u[1], self.B[1], self.E)

        self.ghosts_periodic(self.E, 0)

        self.sum_fields_u_1_order(self.u[0], self.u[1], dT, self.rho[0])

        self.sum_fields_1_order(self.rho[0], self.rho[1], dT)

        self.div_fields_u_1_order(self.u[0], self.rho[0])

        self.computeB_staggered(self.E, self.B_staggered[1])
        self.ghosts_periodic(self.B_staggered[1], 0)

        self.sum_fields_1_order(self.B_staggered[0], self.B_staggered[1], dT)

        self.ghosts_periodic(self.B_staggered[0], 0)

        self.convert_stag_grid(self.get_Bstag0, out_arr=self.B[0])

        self.sum_fields_1_order(self.B[0], self.B[1], dT)

        self.ghosts_periodic(self.B[0], 0)

        self.computeP(self.p[0], self.get_E)

        self.ghosts_periodic(self.u[0], 0)
        self.ghosts_periodic(self.rho[0], 0)
        self.ghosts_periodic(self.p[0], 0)
