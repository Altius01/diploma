import taichi as ti
from abc import ABC, abstractmethod

from taichi_src.common.types import *
from taichi_src.common.boundaries import *
from taichi_src.common.matrix_ops import *
from taichi_src.common.pointers import *
from taichi_src.spatial_diff.diff_fv import *
import taichi_src.spatial_diff.diff as diff_fd
@ti.data_oriented
class SystemComputer:
    def __init__(self, gamma, Re, Ms, Ma, Rem, delta_hall, 
        ghosts, shape, h, domain_size, ideal=False, hall=False, les=NonHallLES.DNS, dim=3):
        self.h = vec3(h)
        self.shape = shape
        self.ghost = ghosts
        self.domain_size = domain_size

        self.Re = Re
        self.Ms = Ms

        self.Ma = Ma
        self.Rem = Rem

        self.delta_hall = delta_hall

        self.gamma = gamma

        self.ideal = ideal
        self.hall = hall
        self.les = les

        self.filter_size = vec3i([1, 1, 1])
        # self.k = -(1.0/3.0)
        self.k = -1

        self.dimensions = dim
        self.implement_dimension()
        self.rho_computer = RhoCompute(self.h, filter_size=self.filter_size, les=les)
        self.u_computer = MomentumCompute(Re, Ma, gamma, self.h, filter_size=self.filter_size, les=les)
        self.B_computer = BCompute(Rem, delta_hall, self.h, filter_size=self.filter_size, les=les)

    def update_data(self, rho, p, u, B):
        self.u = u
        # self.p = p
        self.B = B
        self.rho = rho

    def implement_dimension(self):
        if self.dimensions == 1:
            self.grad_sc = grad_sc_1D
            self.grad_vec = grad_vec_1D
            self.get_dx_st = get_dx_st_1D
            self.rot_vec = rot_vec_1D
            self.div_vec = div_vec_1D
            self.V_plus_sc = V_plus_sc_1D
            self.V_plus_vec = V_plus_vec_1D

            self.flux_HLLD_right = self.flux_HLLD_right_1D
        elif self.dimensions == 2:
            self.grad_sc = grad_sc_2D
            self.grad_vec = grad_vec_2D
            self.get_dx_st = get_dx_st_2D
            self.rot_vec = rot_vec_2D
            self.div_vec = div_vec_2D
            self.V_plus_sc = V_plus_sc_2D
            self.V_plus_vec = V_plus_vec_2D

            self.flux_HLLD_right = self.flux_HLLD_right_2D
        elif self.dimensions == 3:
            self.grad_sc = grad_sc_3D
            self.grad_vec = grad_vec_3D
            self.get_dx_st = get_dx_st_3D
            self.rot_vec = rot_vec_3D
            self.div_vec = div_vec_3D
            self.V_plus_sc = V_plus_sc_3D
            self.V_plus_vec = V_plus_vec_3D

            self.flux_HLLD_right = self.flux_HLLD_right_3D

    @ti.func
    def Q_rho(self, idx):
        return self.rho[idx]
    
    @ti.func
    def Q_u(self, idx):
        return self.rho[idx]*self.u[idx]
    
    @ti.func
    def Q_B(self, idx):
        return self.B[idx]

    @ti.func
    def V_rho(self, idx):
        return self.rho[idx]
    
    @ti.func
    def V_u(self, idx):
        return self.u[idx]
    
    @ti.func
    def V_B(self, idx):
        return self.B[idx]

    @ti.func
    def grad_rho(self, idx):
        return self.grad_sc(self.V_rho, self.h, idx)
    
    @ti.func
    def grad_U(self, idx):
        return self.grad_vec(self.V_u, self.h, idx)
    
    @ti.func
    def grad_B(self, idx):
        return self.grad_vec(self.V_B, self.h, idx)

    @ti.func
    def rot_U(self, idx):
        return self.rot_vec(self.V_u, self.h, idx)
    
    @ti.func
    def rot_B(self, idx):
        return self.rot_vec(self.V_B, self.h, idx)

    @ti.func
    def _check_ghost(self, shape, ghost, idx):
        return (idx < ghost) or (idx >= shape - ghost)

    @ti.func
    def check_ghost_idx(self, idx):
        result = False

        result = result or self._check_ghost(self.shape[0], self.ghost[0], idx[0])
        result = result or self._check_ghost(self.shape[1], self.ghost[1], idx[1])
        result = result or self._check_ghost(self.shape[2], self.ghost[2], idx[2])

        return result

    @ti.func
    def get_eigenvals(self, j, idx):
        u = self.u[idx]
        B = self.B[idx]
        rho = self.rho[idx]

        # pi_rho = ti.sqrt(4 * ti.math.pi * rho)
        pi_rho = ti.sqrt(rho)

        b = (1.0 / self.Ma) * B.norm() / pi_rho
        b_x = (1.0 / self.Ma) * B[j] / pi_rho
        # Sound speed
        _p = self.u_computer.get_pressure(rho)
        c = (1.0 / self.Ms)
        # c = (1.0 / self.Ms) * ti.sqrt(self.gamma * _p / rho)
        # Alfen speed
        c_a = (1.0 / self.Ma) * B[j] / pi_rho

        sq_root = ti.sqrt((b**2 + c**2)**2 - 4 * b_x**2 * c**2)

        # Magnetosonic wawes
        c_s = ti.sqrt( 0.5 * (b**2 + c**2) - sq_root)
        c_f = ti.sqrt( 0.5 * (b**2 + c**2) + sq_root)

        return ti.abs(u[j]) + c_f
        # return ti.Vector([
        #     u[j] + c_f,
        #     u[j] - c_f,
        #     u[j] + c_s,
        #     u[j] - c_s,
        #     u[j] + c_a,
        #     u[j] - c_a,
        #     0,
        # ])

    
    @ti.func
    def get_c_fast(self, Q_rho, Q_u, Q_B, j):
        # pi_rho = ti.sqrt(4 * ti.math.pi * Q_rho)
        pi_rho = ti.sqrt(Q_rho)

        b = (1.0 / self.Ma) * Q_B.norm() / pi_rho
        b_x = (1.0 / self.Ma) * Q_B[j] / pi_rho
        # Sound speed
        _p = self.u_computer.get_pressure(Q_rho)
        c = (1.0 / self.Ms)
        # c = (1.0 / self.Ms) * ti.sqrt(self.gamma * _p / Q_rho)
        # Alfen speed
        c_a = (1.0 / self.Ma) * Q_B[j] / pi_rho

        sq_root = ti.sqrt((b**2 + c**2)**2 - 4 * b_x**2 * c**2)

        # Magnetosonic wawes
        c_f = ti.sqrt( 0.5 * (b**2 + c**2) + sq_root)

        return c_f

    @ti.func
    def get_s_max(self, idx):
        result = vec3(0)

        result[0] = self.get_eigenvals(0, idx)

        if self.dimensions > 1:
            result[1] = self.get_eigenvals(1, idx)
        
        if self.dimensions == 2:
            result[2] = self.get_eigenvals(2, idx)

        return result

    @ti.kernel
    def get_cfl_cond(self) -> vec3:
        result = vec3(0)
        for idx in ti.grouped(self.rho):
            result = ti.max(self.get_s_max(idx), result)

        return result
    
    @ti.func
    def minmod(self, r):
        return ti.max(ti.min(r, 1), 0)
    
    @ti.func
    def Q_L(self, Q: ti.template(), i, idx):
        idx_left = idx - get_basis(i)
        idx_right = idx + get_basis(i)
                                    
        D_m = Q(idx) - Q(idx_left)
        D_p = Q(idx_right) - Q(idx)

        # return Q(idx) + 0.25 * ( (1-self.k)*self.minmod(D_p / D_m)*D_m + (1+self.k)*self.minmod(D_m / D_p)*D_p)
        return Q(idx)

    @ti.func
    def Q_R(self, Q: ti.template(), i, idx):
        idx_left = idx
        idx_right = idx + 2*get_basis(i)
        idx = idx + get_basis(i)

        D_m = Q(idx) - Q(idx_left)
        D_p = Q(idx_right) - Q(idx)

        # return Q(idx) - 0.25 * ( (1+self.k)*self.minmod(D_p / D_m)*D_m + (1-self.k)*self.minmod(D_m / D_p)*D_p)
        return Q(idx)

    @ti.func
    def HLLD(self, flux_rho: ti.template(), flux_u: ti.template(), flux_B: ti.template(), 
        Q_rho_L, Q_u_L, Q_B_L, Q_rho_R, Q_u_R, Q_B_R, i, idx=0, debug=False):

        c_f_L = self.get_c_fast(Q_rho_L, Q_u_L, Q_B_L, i)
        c_f_R = self.get_c_fast(Q_rho_R, Q_u_R, Q_B_R, i)

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

        By_R = Q_B_R[y]
        Bz_R = Q_B_R[z]

        By_L = Q_B_L[y]
        Bz_L = Q_B_L[z]

        s_L = ti.min(u_L - c_f_L, u_R - c_f_R)
        s_R = ti.min(u_L + c_f_L, u_R + c_f_R)

        F_rho_L = get_vec_col(flux_rho(Q_rho_L, Q_u_L, Q_B_L), i)
        F_rho_R = get_vec_col(flux_rho(Q_rho_R, Q_u_R, Q_B_R), i)

        F_u_L = get_mat_col(flux_u(Q_rho_L, Q_u_L, Q_B_L), i)
        F_u_R = get_mat_col(flux_u(Q_rho_R, Q_u_R, Q_B_R), i)

        F_b_L = get_mat_col(flux_B(Q_rho_L, Q_u_L, Q_B_L), i)
        F_b_R = get_mat_col(flux_B(Q_rho_R, Q_u_R, Q_B_R), i)

        Q_rho_hll = (s_R*Q_rho_R - s_L*Q_rho_L - F_rho_R + F_rho_L) / (s_R - s_L)
        F_rho_hll = (s_R*F_rho_L - s_L*F_rho_R + s_R*s_L*(Q_rho_R - Q_rho_L)) / (s_R - s_L)

        Q_u_hll = (s_R*Q_u_R - s_L*Q_u_L - F_u_R + F_u_L) / (s_R - s_L)
        F_u_hll = (s_R*F_u_L - s_L*F_u_R + s_R*s_L*(Q_u_R - Q_u_L)) / (s_R - s_L)

        Q_B_hll = (s_R*Q_B_R - s_L*Q_B_L - F_b_R + F_b_L) / (s_R - s_L)

        Bx = Q_B_hll[x]

        u_star = F_rho_hll / Q_rho_hll

        # ca = (1.0 / self.Ma) * ( ti.abs(Bx) / ti.sqrt(4*ti.math.pi*Q_rho_hll) )
        ca = (1.0 / self.Ma) * ( ti.abs(Bx) / ti.sqrt(Q_rho_hll) )
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

        if ti.abs((s_L - s_L_star)*(s_L - s_R_star)) > 1e-8 and ti.abs((s_R - s_L_star)*(s_R - s_R_star)) > 1e-8:
            rho_v_L_star = Q_rho_hll * v_L - (1.0 / self.Ma**2) * Bx*By_L * (u_star - u_L) / ((s_L - s_L_star)*(s_L - s_R_star))
            rho_v_R_star = Q_rho_hll * v_R - (1.0 / self.Ma**2) * Bx*By_R * (u_star - u_R) / ((s_R - s_L_star)*(s_R - s_R_star))

            rho_w_L_star = Q_rho_hll * w_L - (1.0 / self.Ma**2) * Bx*Bz_L * (u_star - u_L) / ((s_L - s_L_star)*(s_L - s_R_star))
            rho_w_R_star = Q_rho_hll * w_R - (1.0 / self.Ma**2) * Bx*Bz_R * (u_star - u_R) / ((s_R - s_L_star)*(s_R - s_R_star))

            By_L_star = (By_L / Q_rho_hll) *( (Q_rho_L*(s_L - u_L)**2 - (1.0 / self.Ma**2) * Bx**2) / ((s_L - s_L_star)*(s_L - s_R_star)))
            By_R_star = (By_R / Q_rho_hll) *( (Q_rho_R*(s_R - u_R)**2 - (1.0 / self.Ma**2) * Bx**2) / ((s_R - s_L_star)*(s_R - s_R_star)))

            Bz_L_star = (Bz_L / Q_rho_hll) *( (Q_rho_L*(s_L - u_L)**2 - (1.0 / self.Ma**2) * Bx**2) / ((s_L - s_L_star)*(s_L - s_R_star)))
            Bz_R_star = (Bz_R / Q_rho_hll) *( (Q_rho_R*(s_R - u_R)**2 - (1.0 / self.Ma**2) * Bx**2) / ((s_R - s_L_star)*(s_R - s_R_star)))

        Xi = ti.sqrt(Q_rho_hll)*ti.math.sign(Bx)
        rho_v_C_star = 0.5*(rho_v_L_star + rho_v_R_star) + (0.5/self.Ma**2)*(By_R_star - By_L_star)*Xi
        rho_w_C_star = 0.5*(rho_w_L_star + rho_w_R_star) + (0.5/self.Ma**2)*(Bz_R_star - Bz_L_star)*Xi
        By_C_star = 0.5*(By_L_star + By_R_star) + 0.5*(rho_v_R_star - rho_v_L_star) / (Xi)
        Bz_C_star = 0.5*(Bz_L_star + Bz_R_star) + 0.5*(rho_w_R_star - rho_w_L_star) / (Xi)

        result = mat3x3(0)

        Q_u_L_star = Q_u_hll
        Q_u_L_star[y] = rho_v_L_star
        Q_u_L_star[z] = rho_w_L_star

        Q_u_R_star = Q_u_hll
        Q_u_R_star[y] = rho_v_R_star
        Q_u_R_star[z] = rho_w_R_star

        Q_B_L_star = Q_B_L
        Q_B_L_star[y] = By_L_star
        Q_B_L_star[z] = Bz_L_star

        Q_B_R_star = Q_B_R
        Q_B_R_star[y] = By_R_star
        Q_B_R_star[z] = Bz_R_star

        F_rho_C_star = Q_rho_hll*u_star

        F_u_C_star = vec3(0)
        F_u_C_star[x] = F_u_hll[x]
        F_u_C_star[y] = rho_v_C_star*u_star - Bx*By_C_star
        F_u_C_star[z] = rho_w_C_star*u_star - Bx*Bz_C_star

        F_B_C_star = vec3(0)
        F_B_C_star[y] = By_C_star*u_star - (Bx*rho_v_C_star/Q_rho_hll)
        F_B_C_star[z] = Bz_C_star*u_star - (Bx*rho_w_C_star/Q_rho_hll)
        
        if debug:
            print(idx, s_L, s_L_star, s_R_star, s_R, c_f_L, c_f_R)

        if s_L > 0:
            result[0, 0] = F_rho_L
            result[:, 1] = F_u_L
            result[:, 2] = F_b_L
        elif s_L_star > 0:
            result[0, 0] = F_rho_L + s_L*(Q_rho_hll - Q_rho_L)
            result[:, 1] = F_u_L + s_L*(Q_u_L_star - Q_u_L)
            result[:, 2] = F_b_L + s_L*(Q_B_L_star - Q_B_L)
        elif s_R_star > 0:
            result[0, 0] = F_rho_C_star
            result[:, 1] = F_u_C_star
            result[:, 2] = F_B_C_star
        elif s_R > 0:
            result[0, 0] = F_rho_R + s_R*(Q_rho_hll - Q_rho_R)
            result[:, 1] = F_u_R + s_R*(Q_u_R_star - Q_u_R)
            result[:, 2] = F_b_R + s_R*(Q_B_R_star - Q_B_R)
        else:
            result[0, 0] = F_rho_R
            result[:, 1] = F_u_R
            result[:, 2] = F_b_R

        return result

    @ti.kernel
    def computeHLLD(self, out_rho: ti.template(), out_u: ti.template(), 
        out_B: ti.template(), out_E: ti.template()):
        E_flux_idx = mat3x2i([[1, 2], [0, 2], [0, 1]])
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                res = mat3x3(0)

                F_L = mat3x2(0)
                F_R = mat3x2(0)
                for j in range(ti.static(self.dimensions)):
                    idx_r = idx
                    idx_l = idx - get_basis(j)
            
                    flux_r = self.flux_HLLD_right(j, idx_r)
                    
                    flux_l = self.flux_HLLD_right(j, idx_l)

                    E_idx = E_flux_idx[j, :]

                    F_L[j, 0] = flux_l[E_idx[0], 2]
                    F_L[j, 1] = flux_l[E_idx[1], 2]

                    F_R[j, 0] = flux_r[E_idx[0], 2]
                    F_R[j, 1] = flux_r[E_idx[1], 2]

                    if idx[0] > 250 and idx[0] < 256:
                        print("idx: ", idx, "flux_l: ", flux_l[:, 2], " flux_r: ", flux_r[:, 2])

                    res -= (flux_r - flux_l) / get_elem_1d(self.h, j)

                out_rho[idx] = res[0, 0]
                out_u[idx] = res[:, 1]
                out_B[idx] = res[:, 2]

                if idx[0] > 250 and idx[0] < 256:
                    print("idx: ", idx, "Final flux: ", res[:, 2])

                E = vec3(0)
                E[0] = 0.25 * (F_R[2, 1] + F_L[2, 1] - F_L[1, 1] - F_R[1, 1])
                E[1] = 0.25 * (F_R[0, 1] + F_L[0, 1] - F_L[2, 0] - F_R[2, 0])
                E[2] = 0.25 * (F_R[1, 0] + F_L[1, 0] - F_L[0, 0] - F_R[0, 0])
                out_E[idx] = E

    @ti.kernel
    def computeB_staggered(self, E: ti.template(), B_stag_out: ti.template()):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):

                i, j, k = idx

                ijk = vec3i([i, j, k])
                ijm1k = vec3i([i, j-1, k])
                ijkm1 = vec3i([i, j, k-1])
                ijm1km1 = vec3i([i, j-1, k-1])
                im1jm1k = vec3i([i-1, j-1, k])
                im1jk = vec3i([i-1, j, k])                

                res = vec3(0)
                res[0] = (
                    self.h[2]*(E[ijk][2] - E[ijm1k][2]) 
                    + self.h[1]*(E[ijkm1][1] - E[ijk][1])
                    ) / (self.h[1]*self.h[2])

                res[1] = (
                    self.h[0]*(E[ijm1k][0] - E[ijm1km1][0])
                    + self.h[2]*(E[im1jm1k][2] - E[ijm1k][2])
                ) / (self.h[0]*self.h[2])

                res[2] = (
                    self.h[0]*(E[ijm1k][0] - E[ijk][0])
                    + self.h[1]*(E[ijk][1] - E[im1jk][1])
                ) / (self.h[0]*self.h[1])

                B_stag_out[idx] = res

    @ti.func
    def flux_HLLD_right_3D(self, i, idx):
        Q_rho_R = self.Q_R(self.Q_rho, i, idx)
        Q_rho_L = self.Q_L(self.Q_rho, i, idx)

        Q_u_R = self.Q_R(self.Q_u, i, idx)
        Q_u_L = self.Q_L(self.Q_u, i, idx)

        Q_B_R = self.Q_R(self.Q_B, i, idx)
        Q_B_L = self.Q_L(self.Q_B, i, idx)

        result = self.HLLD(self.rho_computer.flux_convective, 
            self.u_computer.flux_convective, 
            self.B_computer.flux_convective,
            Q_rho_L, Q_u_L, Q_B_L, Q_rho_R, Q_u_R, Q_B_R, i)

        for j, k in ti.ndrange(2, 2):
            corner = idx - vec3i(1) + self.get_dx_st(i, j, k, left=False)
            V_rho = self.V_plus_sc(self.V_rho, corner)
            V_u = self.V_plus_vec(self.V_B, corner)
            V_B = self.V_plus_vec(self.V_B, corner)

            if ti.static(self.ideal==False):
                gradU = self.grad_U(corner)
                gradB = self.grad_B(corner)

                result[:, 1] -= 0.25*get_mat_col(
                    self.u_computer.flux_viscous(V_rho, V_u, V_B, gradU, gradB)
                    , i)
                
                result[:, 2] -= 0.25*get_mat_col(
                    self.B_computer.flux_viscous(V_rho, V_u, V_B, gradU, gradB)
                    , i)

            if ti.static(self.hall):
                result[:, 2] -= 0.25*get_mat_col(
                    self.B_computer.flux_hall(V_rho, V_u, V_B, self.grad_B(corner), self.rot_B(corner))
                    , i)
                
            if ti.static(self.les != NonHallLES.DNS):
                gradU = self.grad_U(corner)
                gradB = self.grad_B(corner)
                rotU = self.rot_U(corner)
                rotB = self.rot_B(corner)

                result[:, 1] -= 0.25*get_mat_col(
                    self.u_computer.flux_les(V_rho, V_u, V_B, gradU, gradB, rotU, rotB)
                    , i)

                result[:, 2] -= 0.25*get_mat_col(
                    self.B_computer.flux_les(V_rho, V_u, V_B, gradU, gradB, rotU, rotB)
                    , i)
        
        return result
    
    @ti.func
    def flux_HLLD_right_2D(self, i, idx):
        Q_rho_R = self.Q_R(self.Q_rho, i, idx)
        Q_rho_L = self.Q_L(self.Q_rho, i, idx)

        Q_u_R = self.Q_R(self.Q_u, i, idx)
        Q_u_L = self.Q_L(self.Q_u, i, idx)

        Q_B_R = self.Q_R(self.Q_B, i, idx)
        Q_B_L = self.Q_L(self.Q_B, i, idx)

        result = self.HLLD(self.rho_computer.flux_convective, 
            self.u_computer.flux_convective, 
            self.B_computer.flux_convective,
            Q_rho_L, Q_u_L, Q_B_L, Q_rho_R, Q_u_R, Q_B_R, i)

        for j in ti.ndrange(2):
            corner = idx - vec3i([1, 1, 0]) + self.get_dx_st(i, j, 0, left=False)
            V_rho = self.V_plus_sc(self.V_rho, corner)
            V_u = self.V_plus_vec(self.V_B, corner)
            V_B = self.V_plus_vec(self.V_B, corner)

            if ti.static(self.ideal==False):
                gradU = self.grad_U(corner)
                gradB = self.grad_B(corner)

                result[:, 1] -= 0.25*get_mat_col(
                    self.u_computer.flux_viscous(V_rho, V_u, V_B, gradU, gradB)
                    , i)
                
                result[:, 2] -= 0.25*get_mat_col(
                    self.B_computer.flux_viscous(V_rho, V_u, V_B, gradU, gradB)
                    , i)

            if ti.static(self.hall):
                result[:, 2] -= 0.25*get_mat_col(
                    self.B_computer.flux_hall(V_rho, V_u, V_B, self.grad_B(corner), self.rot_B(corner))
                    , i)
                
            if ti.static(self.les != NonHallLES.DNS):
                gradU = self.grad_U(corner)
                gradB = self.grad_B(corner)
                rotU = self.rot_U(corner)
                rotB = self.rot_B(corner)

                result[:, 1] -= 0.25*get_mat_col(
                    self.u_computer.flux_les(V_rho, V_u, V_B, gradU, gradB, rotU, rotB)
                    , i)

                result[:, 2] -= 0.25*get_mat_col(
                    self.B_computer.flux_les(V_rho, V_u, V_B, gradU, gradB, rotU, rotB)
                    , i)
        
        return result
    
    @ti.func
    def flux_HLLD_right_1D(self, i, idx):
        Q_rho_R = self.Q_R(self.Q_rho, i, idx)
        Q_rho_L = self.Q_L(self.Q_rho, i, idx)

        Q_u_R = self.Q_R(self.Q_u, i, idx)
        Q_u_L = self.Q_L(self.Q_u, i, idx)

        Q_B_R = self.Q_R(self.Q_B, i, idx)
        Q_B_L = self.Q_L(self.Q_B, i, idx)

        debug = idx[0] > 250 and idx[0] < 256
        result = self.HLLD(self.rho_computer.flux_convective, 
            self.u_computer.flux_convective, 
            self.B_computer.flux_convective,
            Q_rho_L, Q_u_L, Q_B_L, Q_rho_R, Q_u_R, Q_B_R, i, idx[0], debug)


        # corner = idx - vec3i([1, 0, 0]) + self.get_dx_st(i, 0, 0, left=False)
        # V_rho = self.V_plus_sc(self.V_rho, corner)
        # V_u = self.V_plus_vec(self.V_B, corner)
        # V_B = self.V_plus_vec(self.V_B, corner)

        # if ti.static(self.ideal==False):
        #     gradU = self.grad_U(corner)
        #     gradB = self.grad_B(corner)

        #     result[:, 1] -= 0.25*get_mat_col(
        #         self.u_computer.flux_viscous(V_rho, V_u, V_B, gradU, gradB)
        #         , i)
            
        #     result[:, 2] -= 0.25*get_mat_col(
        #         self.B_computer.flux_viscous(V_rho, V_u, V_B, gradU, gradB)
        #         , i)

        # if ti.static(self.hall):
        #     result[:, 2] -= 0.25*get_mat_col(
        #         self.B_computer.flux_hall(V_rho, V_u, V_B, self.grad_B(corner), self.rot_B(corner))
        #         , i)
            
        # if ti.static(self.les != NonHallLES.DNS):
        #     gradU = self.grad_U(corner)
        #     gradB = self.grad_B(corner)
        #     rotU = self.rot_U(corner)
        #     rotB = self.rot_B(corner)

        #     result[:, 1] -= 0.25*get_mat_col(
        #         self.u_computer.flux_les(V_rho, V_u, V_B, gradU, gradB, rotU, rotB)
        #         , i)

        #     result[:, 2] -= 0.25*get_mat_col(
        #         self.B_computer.flux_les(V_rho, V_u, V_B, gradU, gradB, rotU, rotB)
        #         , i)
        
        return result

    @ti.kernel
    def computeP(self, out: ti.template(), B: ti.template()):
        for idx in ti.grouped(B):
            if not self.check_ghost_idx(idx):
                # out[idx] = ti.math.pow(ti.cast(rho_new[idx], ti.f32), ti.cast(self.gamma, ti.f32))
                # out[idx] = 0.5*B[idx].norm_sqr()

                out[idx] = self.div_vec(self.V_B, self.h, idx)

    @ti.kernel
    def compute_U(self, rho_u: ti.template(), rho: ti.template()):
        for idx in ti.grouped(rho):
            rho_u[idx] /= rho[idx]

    @ti.kernel    
    def ghosts_periodic_foo(self, out: ti.template()):
        for idx in ti.grouped(out):
            if self.check_ghost_idx(idx):
                out[idx] = out[get_ghost_new_idx(self.ghost, self.shape, idx)]

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
            i, j, k = idx
            if i < self.ghost[0]:
                i = 2*self.ghost[0] - (i+1)
            
            if i >= self.shape[0] - self.ghost[0]:
                i = 2*(self.shape[0]-self.ghost[0]) - (i+1)

            if j < self.ghost[1]:
                j = 2*self.ghost[1] - (j+1)

            if j >= self.shape[1] - self.ghost[1]:
                j = 2*(self.shape[1]-self.ghost[1]) - (j+1)

            if k < self.ghost[2]:
                k = 2*self.ghost[2] - (k+1)

            if k >= self.shape[2] - self.ghost[2]:
                k = 2*(self.shape[2]-self.ghost[2]) - (k+1)

            idx_new = vec3i([i, j, k])

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


@ti.data_oriented
class Compute(ABC):
    def __init__(self, h, filter_size=vec3i(1), les=NonHallLES.DNS):
        self.h = h
        self.filter_size = filter_size
        self.filter_delta = self.h * self.filter_size
        self.les = les

    @ti.func
    @abstractmethod
    def flux_convective(self, Q_rho, Q_u, Q_B):
        ...

    @ti.func
    @abstractmethod
    def flux_viscous(self, V_rho, V_u, V_B, grad_U, grad_B):
        ...

    @ti.func
    @abstractmethod
    def flux_hall(self, V_rho, V_u, V_B, grad_B, rot_B):
        ...

    @ti.func
    @abstractmethod
    def flux_les(self, V_rho, V_u, V_B, grad_U, grad_B, rot_U, rot_B):
        ...


@ti.data_oriented
class RhoCompute(Compute):
    @ti.func
    def flux_convective(self, Q_rho, Q_u, Q_B):
        return Q_u

    @ti.func
    def flux_viscous(self, V_rho, V_u, V_B, grad_U, grad_B):
        return vec3(0)

    @ti.func
    def flux_hall(self, V_rho, V_u, V_B, grad_B, rot_B):
        return vec3(0)

    @ti.func
    def flux_les(self, V_rho, V_u, V_B, grad_U, grad_B, rot_U, rot_B):
        return vec3(0)


@ti.data_oriented
class MomentumCompute(Compute):
    def __init__(self, Re, Ma, gamma, h, filter_size=vec3i(1), les=NonHallLES.DNS):
        super().__init__(h, filter_size=filter_size, les=les)
        self.Re = Re
        self.Ma = Ma
        self.gamma = gamma

    @ti.func
    def get_pressure(self, rho):
        # return ti.pow(rho, self.gamma)
        return rho

    @ti.func
    def flux_convective(self, Q_rho, Q_u, Q_B):
        p = self.get_pressure(Q_rho)
        BB = Q_B.outer_product(Q_B)
        rho_UU = Q_u.outer_product(Q_u) / Q_rho
        return (
            rho_UU
            + (p + (0.5/self.Ma**2) * Q_B.norm_sqr()) * kron
            - BB
        )

    @ti.func
    def flux_viscous(self, V_rho, V_u, V_B, grad_U, grad_B):
        divU = grad_U.trace()

        return (
            grad_U + grad_U.transpose() + (2.0/3.0) * divU * kron
        ) / self.Re
    
    @ti.func
    def flux_hall(self, V_rho, V_u, V_B, grad_B, rot_B):
        return mat3x3(0)

    @ti.func
    def flux_les(self, V_rho, V_u, V_B, grad_U, grad_B, rot_U, rot_B):
        return mat3x3(0)


@ti.data_oriented
class BCompute(Compute):
    def __init__(self, Rem, delta_hall, h, filter_size=vec3i(1), les=NonHallLES.DNS):
        super().__init__(h, filter_size=filter_size, les=les)
        self.Rem = Rem
        self.delta_hall = delta_hall

    @ti.func
    def flux_convective(self, Q_rho, Q_u, Q_B):
        Bu = Q_B.outer_product(Q_u) / Q_rho
        return Bu - Bu.transpose()

    @ti.func
    def flux_viscous(self, V_rho, V_u, V_B, grad_U, grad_B):
        return (
            grad_B - grad_B.transpose()
        ) / self.Rem

    @ti.func
    def flux_hall(self, V_rho, V_u, V_B, grad_B, rot_B):
        j = rot_B
        v_h = - self.delta_hall * j / V_rho
        v_hB = v_h.outer_product(V_B)  
        return (
            v_hB - v_hB.transpose()
        )

    @ti.func
    def flux_les(self, V_rho, V_u, V_B, grad_U, grad_B, rot_U, rot_B):
        return mat3x3(0)
