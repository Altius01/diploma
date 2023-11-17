from abc import ABC, abstractmethod

import taichi as ti
from taichi_src.common.matrix_ops import get_idx_to_basis, get_mat_col, get_vec_col

from taichi_src.common.types import double, vec3, mat3x3, kron


@ti.data_oriented
class FluxMHD(ABC):
    def __init__(self, Q_rho, Q_momentum, Q_magnetic, Ma) -> None:
        self.Q_rho = Q_rho
        self.Q_momentum = Q_momentum
        self.Q_magnetic = Q_magnetic

        self.Ma = Ma

    @ti.func
    @abstractmethod
    def flux_right(self, idx):
        raise NotImplementedError()

    @ti.func
    @abstractmethod
    def pressure(self, idx):
        raise NotImplementedError()

    def F_rho(self, idx):
        return self.Q_momentum(idx)

    def F_momentum(self, idx):
        p = self.pressure(self.Q_rho(idx))
        BB = self.Q_magnetic(idx).outer_product(self.Q_magnetic(idx))
        rho_UU = self.Q_momentum(idx).outer_product(self.Q_momentum(idx)) / self.Q_rho(
            idx
        )
        return (
            rho_UU
            + (p + (0.5 / self.Ma**2) * self.Q_magnetic(idx).norm_sqr()) * kron
            - BB
        )

    def F_magnetic(self, idx):
        Bu = self.Q_magnetic(idx).outer_product(self.Q_momentum(idx)) / self.Q_rho(idx)
        return Bu - Bu.transpose()


@ti.data_oriented
class FluxMHDPolytropical(FluxMHD):
    def __init__(self, gamma, Q_rho, Q_momentum, Q_magnetic, Ma) -> None:
        self.gamma = gamma
        super().__init__(Q_rho, Q_momentum, Q_magnetic, Ma)

    @ti.func
    def pressure(self, idx):
        return ti.pow(self.Q_rho(idx), self.gamma)


@ti.data_oriented
class PolytropicalHLLD(FluxMHDPolytropical):
    @ti.func
    def get_sound_speed(self, p, rho):
        return (1.0 / self.Ms) * ti.sqrt(self.gamma * p / rho)

    @ti.func
    def get_c_fast(self, idx):
        inv_pi_rho = 1.0 / ti.sqrt(self.Q_rho(idx))
        inv_Ma = 1.0 / self.Ma
        b = inv_Ma * self.Q_magnetic(idx).norm() * inv_pi_rho
        b_x = inv_Ma * self.Q_magnetic(idx) * inv_pi_rho

        # Sound speed
        _p = self.pressure(idx)
        c = self.get_sound_speed(_p, self.Q_rho(idx))

        sq_root = ti.sqrt((b**2 + c**2) ** 2 - 4 * b_x**2 * c**2)

        # Magnetosonic wawes
        c_f = ti.sqrt(0.5 * ((b**2 + c**2) + sq_root))
        return c_f

    @ti.func
    def HLLD(
        self,
        Q_rho_L,
        Q_u_L,
        Q_b_L,
        Q_rho_R,
        Q_u_R,
        Q_b_R,
        i,
    ):
        c_f_L = self.get_c_fast(Q_rho_L, Q_u_L, Q_b_L, i)
        c_f_R = self.get_c_fast(Q_rho_R, Q_u_R, Q_b_R, i)

        yz = get_idx_to_basis(i)
        x = i
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
        s_R = ti.max(u_L + c_f_L, u_R + c_f_R)

        F_rho_L = get_vec_col(self.F_rho(Q_rho_L, Q_u_L, Q_b_L), i)
        F_rho_R = get_vec_col(self.F_rho(Q_rho_R, Q_u_R, Q_b_R), i)

        F_u_L = get_mat_col(self.F_momentum(Q_rho_L, Q_u_L, Q_b_L), i)
        F_u_R = get_mat_col(self.F_momentum(Q_rho_R, Q_u_R, Q_b_R), i)

        F_b_L = get_mat_col(self.F_magnetic(Q_rho_L, Q_u_L, Q_b_L), i)
        F_b_R = get_mat_col(self.F_magnetic(Q_rho_R, Q_u_R, Q_b_R), i)

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

        ca = (1.0 / self.Ma) * (ti.abs(Bx) / ti.sqrt(Q_rho_hll))
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
            rho_v_L_star = Q_rho_hll * v_L - (1.0 / self.Ma**2) * Bx * By_L * (
                u_star - u_L
            ) / ((s_L - s_L_star) * (s_L - s_R_star))
            rho_v_R_star = Q_rho_hll * v_R - (1.0 / self.Ma**2) * Bx * By_R * (
                u_star - u_R
            ) / ((s_R - s_L_star) * (s_R - s_R_star))

            rho_w_L_star = Q_rho_hll * w_L - (1.0 / self.Ma**2) * Bx * Bz_L * (
                u_star - u_L
            ) / ((s_L - s_L_star) * (s_L - s_R_star))
            rho_w_R_star = Q_rho_hll * w_R - (1.0 / self.Ma**2) * Bx * Bz_R * (
                u_star - u_R
            ) / ((s_R - s_L_star) * (s_R - s_R_star))

            By_L_star = (By_L / Q_rho_hll) * (
                (Q_rho_L * (s_L - u_L) ** 2 - (1.0 / self.Ma**2) * Bx**2)
                / ((s_L - s_L_star) * (s_L - s_R_star))
            )
            By_R_star = (By_R / Q_rho_hll) * (
                (Q_rho_R * (s_R - u_R) ** 2 - (1.0 / self.Ma**2) * Bx**2)
                / ((s_R - s_L_star) * (s_R - s_R_star))
            )

            Bz_L_star = (Bz_L / Q_rho_hll) * (
                (Q_rho_L * (s_L - u_L) ** 2 - (1.0 / self.Ma**2) * Bx**2)
                / ((s_L - s_L_star) * (s_L - s_R_star))
            )
            Bz_R_star = (Bz_R / Q_rho_hll) * (
                (Q_rho_R * (s_R - u_R) ** 2 - (1.0 / self.Ma**2) * Bx**2)
                / ((s_R - s_L_star) * (s_R - s_R_star))
            )

        Xi = ti.sqrt(Q_rho_hll) * ti.math.sign(Bx)

        rho_v_C_star = (
            0.5 * (rho_v_L_star + rho_v_R_star)
            + (0.5 / self.Ma**2) * (By_R_star - By_L_star) * Xi
        )
        rho_w_C_star = (
            0.5 * (rho_w_L_star + rho_w_R_star)
            + (0.5 / self.Ma**2) * (Bz_R_star - Bz_L_star) * Xi
        )
        By_C_star = 0.5 * (By_L_star + By_R_star) + 0.5 * (
            (rho_v_R_star - rho_v_L_star) / Xi
        )
        Bz_C_star = 0.5 * (Bz_L_star + Bz_R_star) + 0.5 * (
            (rho_w_R_star - rho_w_L_star) / Xi
        )

        result = mat3x3(0)

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
            result[0, 0] = F_rho_L
            result[:, 1] = F_u_L
            result[:, 2] = F_b_L
        elif s_L_star > 0:
            result[0, 0] = F_rho_L + s_L * (Q_rho_hll - Q_rho_L)
            result[:, 1] = F_u_L + s_L * (Q_u_L_star - Q_u_L)
            result[:, 2] = F_b_L + s_L * (Q_b_L_star - Q_b_L)
        elif s_R_star > 0:
            result[0, 0] = F_rho_C_star
            result[:, 1] = F_u_C_star
            result[:, 2] = F_B_C_star
        elif s_R > 0:
            result[0, 0] = F_rho_R + s_R * (Q_rho_hll - Q_rho_R)
            result[:, 1] = F_u_R + s_R * (Q_u_R_star - Q_u_R)
            result[:, 2] = F_b_R + s_R * (Q_b_R_star - Q_b_R)
        else:
            result[0, 0] = F_rho_R
            result[:, 1] = F_u_R
            result[:, 2] = F_b_R

        return result
