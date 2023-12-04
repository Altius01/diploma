from abc import ABC, abstractmethod

import taichi as ti

from src.common.matrix_ops import get_idx_to_basis, get_mat_col, get_vec_col
from src.flux.flux import MagneticFlux, MomentumFlux, RhoFlux, parse_vars

from src.common.types import *


@ti.data_oriented
class Solver(ABC):
    def __init__(
        self, flux_rho: RhoFlux, flux_u: MomentumFlux, flux_b: MagneticFlux, axes=0
    ):
        self.axes = 0
        self.flux_rho: RhoFlux = flux_rho
        self.flux_u: MomentumFlux = flux_u
        self.flux_b: MagneticFlux = flux_b

    @ti.func
    @abstractmethod
    def get_conv(self, q_l: vec7, q_r: vec7):
        raise NotImplementedError

    @ti.func
    def get_viscous(self, v, grad_U, grad_B):
        viscous_flux = vec7(0)
        viscous_flux[1:4] = self.flux_u.flux_viscous(v, grad_U, grad_B)
        viscous_flux[4:] = self.flux_b.flux_viscous(v, grad_U, grad_B)

        return viscous_flux

    @ti.func
    def get_hall(self, v, grad_B, rot_B):
        hall_flux = vec7(0)
        hall_flux[4:] = self.flux_b.flux_hall(v, grad_B, rot_B)

        return hall_flux

    @ti.func
    def get_les(self, v, grad_U, grad_B, rot_U, rot_B):
        les_flux = vec7(0)
        les_flux[1:4] = self.flux_u.flux_les(v, grad_U, grad_B, rot_U, rot_B)
        les_flux[4:] = self.flux_b.flux_les(v, grad_U, grad_B, rot_U, rot_B)

        return les_flux

    @ti.func
    def get_c_fast(self, q):
        q_rho, q_u, q_b = parse_vars(q)

        pi_rho = ti.sqrt(q_rho)

        b = (1.0 / self.flux_u.Ma) * (q_b.norm() / pi_rho)
        b_x = (1.0 / self.flux_u.Ma) * (q_b[self.axes] / pi_rho)

        # Sound speed
        _p = self.flux_u.get_pressure(q_rho)
        c = self.flux_u.get_sound_speed(_p, q_rho)

        sq_root = ti.sqrt((b**2 + c**2) ** 2 - 4 * b_x**2 * c**2)

        # Magnetosonic wawes
        c_f = ti.sqrt(0.5 * ((b**2 + c**2) + sq_root))

        return c_f

    @ti.func
    def get_max_eigenval(self, q):
        q_rho, q_u, q_b = parse_vars(q)

        # pi_rho = ti.sqrt(4 * ti.math.pi * q_rho)
        pi_rho = ti.sqrt(q_rho)

        b = (1.0 / self.flux_u.Ma) * q_b.norm() / pi_rho
        b_x = (1.0 / self.flux_u.Ma) * q_b[self.axes] / pi_rho
        # Sound speed
        _p = self.flux_u.get_pressure(q_rho)
        c = self.flux_u.get_sound_speed(_p, q_rho)
        # Alfen speed
        c_a = (1.0 / self.flux_u.Ma) * q_b[self.axes] / pi_rho

        sq_root = ti.sqrt((b**2 + c**2) ** 2 - 4 * b_x**2 * c**2)

        # Magnetosonic wawes
        c_s = ti.sqrt(0.5 * (b**2 + c**2) - sq_root)
        c_f = ti.sqrt(0.5 * (b**2 + c**2) + sq_root)

        return ti.max(
            ti.abs(q_u[self.axes]) + c_f,
            ti.abs(q_u[self.axes]) + c_s,
            ti.abs(q_u[self.axes]) + c_a,
        )

        # return ti.Vector(
        #     [
        #         q_u[self.axes] + c_f,
        #         q_u[self.axes] - c_f,
        #         q_u[self.axes] + c_s,
        #         q_u[self.axes] - c_s,
        #         q_u[self.axes] + c_a,
        #         q_u[self.axes] - c_a,
        #         0,
        #     ]
        # )


@ti.data_oriented
class RoeSolver(Solver):
    @ti.func
    def get_conv(self, q_l: vec7, q_r: vec7):
        conv_flux = vec7(0)

        s_max = ti.max(
            self.get_max_eigenval(q_l),
            self.get_max_eigenval(q_r),
        )

        conv_flux[0] = get_vec_col(
            self.flux_rho.flux_convective(q_l) + self.flux_rho.flux_convective(q_r),
            self.axes,
        )

        conv_flux[1:4] = get_mat_col(
            self.flux_u.flux_convective(q_l) + self.flux_u.flux_convective(q_r),
            self.axes,
        )

        conv_flux[4:] = get_mat_col(
            self.flux_b.flux_convective(q_l) + self.flux_b.flux_convective(q_r),
            self.axes,
        )

        conv_flux -= s_max * (q_r - q_l)
        return 0.5 * conv_flux


@ti.data_oriented
class HLLDSolver(Solver):
    @ti.func
    def get_conv(self, q_l: vec7, q_r: vec7):
        Q_rho_L = q_l[0]
        Q_u_L = q_l[1:4]
        Q_b_L = q_l[4:]

        Q_rho_R = q_r[0]
        Q_u_R = q_r[1:4]
        Q_b_R = q_r[4:]

        c_f_L = self.get_c_fast(q_l)
        c_f_R = self.get_c_fast(q_r)

        yz = get_idx_to_basis(self.axes)
        x = self.axes
        y = yz[0]
        z = yz[1]

        u_R = Q_u_R[x] / Q_rho_R
        v_R = Q_u_R[y] / Q_rho_R
        w_R = Q_u_R[z] / Q_rho_R

        u_L = Q_u_L[x] / Q_rho_L
        v_L = Q_u_L[y] / Q_rho_L
        w_L = Q_u_L[z] / Q_rho_L

        By_R = Q_b_R[y]
        Bz_R = Q_b_R[z]

        By_L = Q_b_L[y]
        Bz_L = Q_b_L[z]

        s_L = ti.min(u_L - c_f_L, u_R - c_f_R)
        s_R = ti.min(u_L + c_f_L, u_R + c_f_R)
        # s_R = ti.max(u_L + c_f_L, u_R + c_f_R)

        F_rho_L = get_vec_col(self.flux_rho.flux_convective(q_l), idx=self.axes)
        F_rho_R = get_vec_col(self.flux_rho.flux_convective(q_r), idx=self.axes)

        F_u_L = get_mat_col(self.flux_u.flux_convective(q_l), idx=self.axes)
        F_u_R = get_mat_col(self.flux_u.flux_convective(q_r), idx=self.axes)

        F_b_L = get_mat_col(self.flux_b.flux_convective(q_l), idx=self.axes)
        F_b_R = get_mat_col(self.flux_b.flux_convective(q_r), idx=self.axes)

        Q_rho_hll = (s_R * Q_rho_R - s_L * Q_rho_L - F_rho_R + F_rho_L) / (s_R - s_L)
        F_rho_hll = (
            s_R * F_rho_L - s_L * F_rho_R + s_R * s_L * (Q_rho_R - Q_rho_L)
        ) / (s_R - s_L)

        Q_u_hll = (s_R * Q_u_R - s_L * Q_u_L - F_u_R + F_u_L) / (s_R - s_L)
        F_u_hll = (s_R * F_u_L - s_L * F_u_R + s_R * s_L * (Q_u_R - Q_u_L)) / (
            s_R - s_L
        )

        Q_b_hll = (s_R * Q_b_R - s_L * Q_b_L - F_b_R + F_b_L) / (s_R - s_L)
        Bx = Q_b_L[x]

        u_star = F_rho_hll / Q_rho_hll

        ca = (1.0 / self.flux_u.Ma) * (ti.abs(Bx) / ti.sqrt(Q_rho_hll))
        s_L_star = u_star - ca
        s_R_star = u_star + ca

        rho_v_L_star = Q_rho_hll * v_L
        rho_v_R_star = Q_rho_hll * v_R

        rho_w_L_star = Q_rho_hll * w_L
        rho_w_R_star = Q_rho_hll * w_R

        By_L_star = double(0.0)
        By_R_star = double(0.0)

        Bz_L_star = double(0.0)
        Bz_R_star = double(0.0)

        if (
            ti.abs((s_L - s_L_star) * (s_L - s_R_star)) > 1e-8
            and ti.abs((s_R - s_L_star) * (s_R - s_R_star)) > 1e-8
        ):
            rho_v_L_star = Q_rho_hll * v_L - (1.0 / self.flux_u.Ma**2) * Bx * By_L * (
                u_star - u_L
            ) / ((s_L - s_L_star) * (s_L - s_R_star))
            rho_v_R_star = Q_rho_hll * v_R - (1.0 / self.flux_u.Ma**2) * Bx * By_R * (
                u_star - u_R
            ) / ((s_R - s_L_star) * (s_R - s_R_star))

            rho_w_L_star = Q_rho_hll * w_L - (1.0 / self.flux_u.Ma**2) * Bx * Bz_L * (
                u_star - u_L
            ) / ((s_L - s_L_star) * (s_L - s_R_star))
            rho_w_R_star = Q_rho_hll * w_R - (1.0 / self.flux_u.Ma**2) * Bx * Bz_R * (
                u_star - u_R
            ) / ((s_R - s_L_star) * (s_R - s_R_star))

            By_L_star = (By_L / Q_rho_hll) * (
                (Q_rho_L * (s_L - u_L) ** 2 - (1.0 / self.flux_u.Ma**2) * Bx**2)
                / ((s_L - s_L_star) * (s_L - s_R_star))
            )
            By_R_star = (By_R / Q_rho_hll) * (
                (Q_rho_R * (s_R - u_R) ** 2 - (1.0 / self.flux_u.Ma**2) * Bx**2)
                / ((s_R - s_L_star) * (s_R - s_R_star))
            )

            Bz_L_star = (Bz_L / Q_rho_hll) * (
                (Q_rho_L * (s_L - u_L) ** 2 - (1.0 / self.flux_u.Ma**2) * Bx**2)
                / ((s_L - s_L_star) * (s_L - s_R_star))
            )
            Bz_R_star = (Bz_R / Q_rho_hll) * (
                (Q_rho_R * (s_R - u_R) ** 2 - (1.0 / self.flux_u.Ma**2) * Bx**2)
                / ((s_R - s_L_star) * (s_R - s_R_star))
            )

        Xi = ti.sqrt(Q_rho_hll) * ti.math.sign(Bx)

        rho_v_C_star = (
            0.5 * (rho_v_L_star + rho_v_R_star)
            + (0.5 / self.flux_u.Ma**2) * (By_R_star - By_L_star) * Xi
        )
        rho_w_C_star = (
            0.5 * (rho_w_L_star + rho_w_R_star)
            + (0.5 / self.flux_u.Ma**2) * (Bz_R_star - Bz_L_star) * Xi
        )
        By_C_star = 0.5 * (By_L_star + By_R_star) + 0.5 * (
            (rho_v_R_star - rho_v_L_star) / Xi
        )
        Bz_C_star = 0.5 * (Bz_L_star + Bz_R_star) + 0.5 * (
            (rho_w_R_star - rho_w_L_star) / Xi
        )

        result = vec7(0)

        Q_u_L_star = Q_u_hll
        Q_u_L_star[y] = rho_v_L_star
        Q_u_L_star[z] = rho_w_L_star

        Q_u_R_star = Q_u_hll
        Q_u_R_star[y] = rho_v_R_star
        Q_u_R_star[z] = rho_w_R_star

        Q_b_L_star = Q_b_L
        Q_b_L_star[y] = By_L_star
        Q_b_L_star[z] = Bz_L_star

        Q_b_R_star = Q_b_R
        Q_b_R_star[y] = By_R_star
        Q_b_R_star[z] = Bz_R_star

        F_rho_C_star = Q_rho_hll * u_star

        F_u_C_star = vec3(0)
        F_u_C_star[x] = F_u_hll[x]
        F_u_C_star[y] = rho_v_C_star * u_star - Bx * By_C_star
        F_u_C_star[z] = rho_w_C_star * u_star - Bx * Bz_C_star

        F_B_C_star = vec3(0)
        F_B_C_star[y] = By_C_star * u_star - (Bx * rho_v_C_star / Q_rho_hll)
        F_B_C_star[z] = Bz_C_star * u_star - (Bx * rho_w_C_star / Q_rho_hll)

        if s_L > 0:
            result[0] = F_rho_L
            result[1:4] = F_u_L
            result[4:] = F_b_L
        elif s_L_star > 0:
            result[0] = F_rho_L + s_L * (Q_rho_hll - Q_rho_L)
            result[1:4] = F_u_L + s_L * (Q_u_L_star - Q_u_L)
            result[4:] = F_b_L + s_L * (Q_b_L_star - Q_b_L)
        elif s_R_star > 0:
            result[0] = F_rho_C_star
            result[1:4] = F_u_C_star
            result[4:] = F_B_C_star
        elif s_R > 0:
            result[0] = F_rho_R + s_R * (Q_rho_hll - Q_rho_R)
            result[1:4] = F_u_R + s_R * (Q_u_R_star - Q_u_R)
            result[4:] = F_b_R + s_R * (Q_b_R_star - Q_b_R)
        else:
            result[0] = F_rho_R
            result[1:4] = F_u_R
            result[4:] = F_b_R

        return result
