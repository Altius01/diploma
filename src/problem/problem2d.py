from dataclasses import dataclass
import taichi as ti
from src.problem.problem import Problem
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


class Problem2D(Problem):
    @ti.kernel
    def get_cfl_cond(self) -> vec3:
        result = vec3(0)

        for idx in ti.grouped(self.rho):
            ti.atomic_max(result, self.get_s_j_max(idx))

        return result

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
                res = vec7(0)
                Flux_E = mat3x3(0)

                for axes in ti.static(range(self.dimensions)):
                    idx_r = idx
                    idx_l = idx - get_basis(axes)

                    q_l = tvd_1order(self.q, idx_r - get_basis(axes))
                    q_r = tvd_1order(self.q, idx_r)

                    flux_r = self.get_flux_right(self.solvers[axes], q_l, q_r)

                    q_l = tvd_1order(self.q, idx_l - get_basis(axes))
                    q_r = tvd_1order(self.q, idx_l)

                    flux_l = self.get_flux_right(
                        self.solvers[axes], self.q(idx), self.q(idx)
                    )

                    flux = (flux_r - flux_l) / get_elem_1d(self.h, axes)

                    res -= flux

                    Flux_E[axes, :] = flux_r[4:]

                out_rho[idx] = res[0]
                out_u[idx] = res[1:4]
                out_B[idx] = res[4:]

                # i = idx[0]
                # j = idx[1]
                # k = idx[2]

                # ijk = idx

                # im1jk = vec3i([i - 1, j, k])
                # ijm1k = vec3i([i, j - 1, k])

                # im1jm1k = vec3i([i - 1, j - 1, k])

                # im1jk = get_ghost_new_idx(self.ghost, self.shape, im1jk)
                # ijm1k = get_ghost_new_idx(self.ghost, self.shape, ijm1k)

                # im1jm1k = get_ghost_new_idx(self.ghost, self.shape, im1jm1k)

                # ti.atomic_add(out_E[ijk][2], 0.25 * (Flux_E[1, 0] - Flux_E[0, 1]))
                # ti.atomic_add(out_E[im1jk][2], 0.25 * (Flux_E[1, 0]))
                # ti.atomic_add(out_E[ijm1k][2], -0.25 * (Flux_E[0, 1]))

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

    @ti.func
    def get_flux_right(self, solver, q_l, q_r, axes=0):
        result = solver.get_conv(q_l, q_r)

        # corner = idx - vec3i([1, 1, 0]) + get_dx_st_2D(self.axes, j, 0, left=False)
        # v_rho = V_plus_sc_2D(self.v_rho, corner)
        # v_u = V_plus_vec_2D(self.v_u, corner)
        # v_b = V_plus_vec_2D(self.v_b, corner)

        # v_corner = vec7(0)

        # v_corner[0] = v_rho
        # v_corner[1:4] = v_u
        # v_corner[4:] = v_b

        # if ti.static(self.ideal == False):
        #     gradU = self.grad_U(corner)
        #     gradB = self.grad_B(corner)

        #     result[1:4] -= 0.5 * get_mat_col(
        #         self.u_computer.flux_viscous(v_corner, gradU, gradB), self.axes
        #     )

        #     result[4:] -= 0.5 * get_mat_col(
        #         self.B_computer.flux_viscous(v_corner, gradU, gradB), self.axes
        #     )

        # if ti.static(self.hall):
        #     result[4:] -= 0.5 * get_mat_col(
        #         self.B_computer.flux_hall(
        #             v_rho, v_u, v_b, self.grad_B(corner), self.rot_B(corner)
        #         ),
        #         self.axes,
        #     )

        # if ti.static(self.les != NonHallLES.DNS):
        #     gradU = self.grad_U(corner)
        #     gradB = self.grad_B(corner)
        #     rotU = self.rot_U(corner)
        #     rotB = self.rot_B(corner)

        #     result[1:4] -= 0.5 * get_mat_col(
        #         self.u_computer.flux_les(v_corner, gradU, gradB, rotU, rotB),
        #         idx=self.axes,
        #     )

        #     result[4:] -= 0.5 * get_mat_col(
        #         self.B_computer.flux_les(v_corner, gradU, gradB, rotU, rotB),
        #         idx=self.axes,
        #     )

        return result

    @ti.kernel
    def computeP(self, out: ti.template(), foo_B: ti.template()):
        for idx in ti.grouped(out):
            if not self.check_ghost_idx(idx):
                out[idx] = self.div_vec(foo_B, self.h, idx)

    @ti.kernel
    def get_speed_from_momentum(self, rho_u: ti.template(), rho: ti.template()):
        for idx in ti.grouped(rho):
            rho_u[idx] /= rho[idx]


class LegacyCompatibleProblem2D(Problem2D):
    @ti.kernel
    def ghosts_periodic_foo(self, out: ti.template()):
        for idx in ti.grouped(out):
            if self.check_ghost_idx(idx):
                out[idx] = out[get_ghost_new_idx(self.ghost, self.shape, idx)]

    @ti.kernel
    def ghosts_mirror_foo(self, out: ti.template()):
        for idx in ti.grouped(out):
            idx_new = get_mirror_new_idx(self.ghost, self.shape, idx)

            if idx_new[0] != idx[0] or idx_new[1] != idx[1] or idx_new[2] != idx[2]:
                out[idx] = out[idx_new]

    def ghosts_mirror_foo_call(self, out):
        self.ghosts_mirror_foo(out)

    def ghosts_periodic_foo_call(self, out):
        self.ghosts_periodic_foo(out)

    @ti.kernel
    def ghosts_periodic(self, rho: ti.template(), u: ti.template(), B: ti.template()):
        for idx in ti.grouped(rho):
            if self.check_ghost_idx(idx):
                new_idx = get_ghost_new_idx(self.ghost, self.shape, idx)
                rho[idx] = rho[new_idx]
                u[idx] = u[new_idx]
                B[idx] = B[new_idx]

    def ghosts_periodic_call(self, rho, u, B):
        self.ghosts_periodic(rho, u, B)

    @ti.kernel
    def ghosts_wall(self, rho: ti.template(), u: ti.template(), B: ti.template()):
        for idx in ti.grouped(rho):
            idx_new = get_mirror_new_idx(self.ghost, self.shape, idx)

            if idx_new[0] != idx[0] or idx_new[1] != idx[1] or idx_new[2] != idx[2]:
                rho[idx] = rho[idx_new]
                u[idx] = -u[idx_new]
                B[idx] = B[idx_new]

    def ghosts_wall_call(self, rho, u, B):
        self.ghosts_wall(rho, u, B)

    @ti.kernel
    def get_foo(self, foo: ti.template(), out: ti.template()):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                out[idx] = foo(idx)

    def get_field_from_foo(self, foo, out):
        self.get_foo(foo, out)
        self.ghosts_periodic_foo_call(out)
        return out

    def get_sc_field_from_foo(self, foo, out):
        # out = ti.field(dtype=double, shape=self.shape)
        return self.get_field_from_foo(foo, out)

    def get_vec_field_from_foo(self, foo, out):
        # out = ti.Vector.field(n=3, dtype=double, shape=self.shape)
        return self.get_field_from_foo(foo, out)

    def get_mat_field_from_foo(self, foo, out):
        # out = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        return self.get_field_from_foo(foo, out)
