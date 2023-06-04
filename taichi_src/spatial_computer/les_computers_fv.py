import taichi as ti

from taichi_src.common.pointers import *
from taichi_src.common.field_ops import *
from taichi_src.filtering.box_filter import *
from taichi_src.spatial_computer.computers_fv import *
    

class LesComputer(SystemComputer):
    def __init__(self, gamma, Re, Ms, Ma, Rem, delta_hall, 
        ghosts, shape, h, domain_size, ideal=False, hall=False, les=NonHallLES.DNS, dim=3):
        super().__init__(gamma, Re, Ms, Ma, Rem, delta_hall, 
        ghosts, shape, h, domain_size, ideal, hall, les, dim)

        self.D = 0
        self.C = 0
        self.Y = 0

        self.u_computer = LesMomentumCompute(Re, Ma, gamma, self.h, les=les)
        self.B_computer = LesBCompute(Rem, delta_hall, self.h, les=les)

        self.init_les_arrays()

        if self.dimensions == 1:
            self._box_filter = box_filter_1D
        elif self.dimensions == 2:
            self._box_filter = box_filter_2D
        elif self.dimensions == 3:
            self._box_filter = box_filter_3D

    def update_data(self, rho, p, u, B):
        self.u = u
        self.p = p
        self.B = B
        self.rho = rho

        self.B_computer.init_data(D=self.D)
        self.u_computer.init_data(C=self.C, Y=self.Y)

    @ti.kernel
    def get_Mu(self, Mu: ti.template(), Mu_a: ti.template(), alpha_hat: ti.template(), Su_hat: ti.template()):
        for idx in ti.grouped(Mu):
            Mu[idx] = alpha_hat[idx] * (Su_hat[idx] - (1.0/3.0)*kron*Su_hat[idx].trace()) - Mu_a[idx]

    @ti.kernel
    def get_Mu_kk(self, Mu_kk: ti.template(), Mu_kk_a: ti.template(), alpha_kk_hat: ti.template(), Su_abs_hat: ti.template()):
        for idx in ti.grouped(Mu_kk):
            Mu_kk[idx] = (
                (alpha_kk_hat[idx] * Su_abs_hat[idx]) 
                - Mu_kk_a[idx]
            )
            
    @ti.kernel
    def get_mB(self, mB: ti.template(), mB_a: ti.template(), phi_hat: ti.template(), J_hat: ti.template()):
        for idx in ti.grouped(mB):
            mB[idx] = phi_hat[idx] * J_hat[idx] - mB_a[idx]

    @ti.kernel
    def get_Lu(self, L: ti.template(), Lu_a: ti.template(),
        Lu_b: ti.template(), rho_hat: ti.template(), 
        rhoU_hat: ti.template(), B_hat: ti.template()):
        for idx in ti.grouped(L):
            L[idx] = (
                 Lu_a[idx] 
                - (rhoU_hat[idx].outer_product(rhoU_hat[idx]) / rho_hat[idx])
                - (1/self.Ma**2)*(Lu_b[idx] - B_hat[idx].outer_product(B_hat[idx]))
            )

    @ti.kernel
    def get_Lb(self, L: ti.template(), Lb_a: ti.template(),
        rho_hat: ti.template(), rhoU_hat: ti.template(), B_hat: ti.template()):
        for idx in ti.grouped(L):
            res = Lb_a[idx] - ( rhoU_hat[idx].outer_product(B_hat[idx]) / rho_hat[idx] )
            L[idx] = res - res.transpose()

    @ti.func
    def get_Su(self, idx):
        return self.rho[idx]*self.Su[idx]

    @ti.func
    def get_Mu_a_arr(self, idx):
        return self.rho[idx]*self.Mu_a[idx]

    @ti.func
    def get_alpha_ij(self, idx):
        return self.alpha_ij[idx]
    
    @ti.func
    def get_phi_arr(self, idx):
        return self.phi[idx]

    @ti.func
    def get_mB_a_arr(self, idx):
        return self.mB_a[idx]

    @ti.func
    def get_J_arr(self, idx):
        return self.J[idx]

    @ti.func
    def get_Su_abs(self, idx):
        return self.rho[idx]*self.Su_abs[idx]

    @ti.func
    def get_Mu_kk_a(self, idx):
        return self.rho[idx]*self.Mu_kk_a[idx]

    @ti.func
    def get_alpha_kk(self, idx):
        return self.alpha_kk[idx]
    
    @ti.func
    def Lu_a(self, idx):
        return self.rho[idx]*self.u[idx].outer_product(self.u[idx])

    @ti.func
    def Lu_b(self, idx):
        return self.B[idx].outer_product(self.B[idx])

    @ti.func
    def Lb_a(self, idx):
        return self.u[idx].outer_product(self.B[idx])

    @ti.func
    def get_S(self, idx):
        grad_U = self.grad_U(idx)
        return 0.5*(grad_U + grad_U.transpose())

    @ti.func
    def get_Mu_a(self, idx):
        S = self.get_S(idx)
        return self.rho[idx] * self.get_alpha(idx)*(S - (1.0/3.0)*S.trace()*kron)

    @ti.func
    def get_alpha(self, idx):
        return self.u_computer.get_alpha(
            self.V_rho(idx), self.V_u(idx), self.V_b(idx), self.grad_U(idx), self.grad_B(idx), self.rot_U(idx), self.rot_B(idx)
        )

    @ti.func
    def get_J(self, idx):
        grad_B = self.grad_B(idx)
        return 0.5*(grad_B - grad_B.transpose())

    @ti.func
    def get_mB_a(self, idx):
        return self.get_phi(idx)*self.get_J(idx)

    @ti.func
    def get_phi(self, idx):
        return self.B_computer.get_phi(
            self.V_rho(idx), self.V_u(idx), self.V_b(idx), self.grad_U(idx), self.grad_B(idx), self.rot_U(idx), self.rot_B(idx)
        )

    @ti.func
    def get_S_abs(self, idx):
        return ti.sqrt(2)*self.get_S(idx).norm()

    @ti.func
    def get_tr_Mu_a(self, idx):
        return self.get_tr_alpha(idx)*self.get_S_abs(idx)

    @ti.func
    def get_tr_alpha(self, idx):
        return self.u_computer.get_tr_alpha(
            self.V_rho(idx), self.V_u(idx), self.V_b(idx), self.grad_U(idx), self.grad_B(idx), self.rot_U(idx), self.rot_B(idx)
        )

    def init_les_arrays(self):
        self.rho_hat = ti.field(dtype=double, shape=self.shape)
        self.rhoU_hat = ti.Vector.field(n=3, dtype=double, shape=self.shape)
        self.B_hat = ti.Vector.field(n=3, dtype=double, shape=self.shape)

        self.Lu_a_hat = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.Lu_b_hat = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        
        
        self.Lu = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)

        self.Lb_a_hat = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.LB = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        
        self.Mu_a_hat = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.Su_hat = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.alpha_ij_hat = ti.field(dtype=double, shape=self.shape)

        self.Mu = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)

        self.phi_hat =  ti.field(dtype=double, shape=self.shape)
        self.mB_a_hat = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.J_hat = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)

        self.mB = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)

        self.Su_abs_hat = ti.field(dtype=double, shape=self.shape)
        self.Mu_kk_a_hat = ti.field(dtype=double, shape=self.shape)
        self.alpha_kk_hat = ti.field(dtype=double, shape=self.shape)

        self.Mu_kk = ti.field(dtype=double, shape=self.shape)

        self.LuM_field = ti.field(dtype=double, shape=self.shape)

        self.MM_field = ti.field(dtype=double, shape=self.shape)

        self.Lkk_field = ti.field(dtype=double, shape=self.shape)

        self.LbmB_field = ti.field(dtype=double, shape=self.shape)

        self.mBmB_field = ti.field(dtype=double, shape=self.shape)

    def box_filter(self, foo, out, eps=2.0):
        self._box_filter(foo, out, self.check_ghost_idx, eps)
        self.ghosts_periodic_foo_call(out)

    def box_favre_filter(self, foo, out, eps=2.0):
        @ti.func
        def foo_favre(idx):
            return self.rho[0][idx]*foo(idx)
        
        self._box_filter(foo, out, self.check_ghost_idx, eps)
        self.ghosts_periodic_foo_call(out)

        favre_filter_divide(out, self.rho_hat)

    def get_les_consts(self):    
        self.box_filter(self.Q_rho, self.rho_hat)
        self.box_filter(self.Q_u, self.rhoU_hat)
        self.box_filter(self.Q_b, self.B_hat)

        self.box_filter(self.Lu_a, self.Lu_a_hat)
        self.box_filter(self.Lu_b, self.Lu_b_hat)
    
        self.get_Lu(self.Lu, self.Lu_a_hat, self.Lu_b_hat, self.rho_hat, self.rhoU_hat, self.B_hat)
        

        self.box_filter(self.Lb_a, self.Lb_a_hat)
        self.get_Lb(self.LB, self.Lb_a_hat, self.rho_hat, self.rhoU_hat, self.B_hat)

        self.box_favre_filter(self.get_S, self.Su_hat)
        self.box_favre_filter(self.get_Mu_a, self.Mu_a_hat)

        self.box_filter(self.get_alpha, self.alpha_ij_hat)
        
        self.get_Mu(self.Mu, self.Mu_a_hat, self.alpha_ij_hat, self.Su_hat)

        self.box_filter(self.get_J, self.J_hat)
        self.box_filter(self.get_mB_a, self.mB_a_hat)
        self.box_filter(self.get_phi, self.phi_hat)

        self.get_mB(self.mB, self.mB_a_hat, self.phi_hat, self.J_hat)

        self.box_favre_filter(self.get_S_abs, self.Su_abs_hat)
        self.box_favre_filter(self.get_tr_Mu_a, self.Mu_kk_a_hat)

        self.box_filter(self.get_tr_alpha, self.alpha_kk_hat)

        self.get_Mu_kk(self.Mu_kk, self.Mu_kk_a_hat, self.alpha_kk_hat, self.Su_abs_hat)
        knl_norm_sqr_field_mat(self.Lu, self.Mu, self.LuM_field)
    
        LuM = Sum.sum_sc_field(self.LuM_field)

        knl_norm_sqr_field_mat(self.Mu, self.Mu, self.MM_field)
        
        MM_mean = Sum.sum_sc_field(self.MM_field)
        knl_tr_field(self.Lu, self.Lkk_field)

        Lkk_mean = Sum.sum_mat_field(self.Lu).trace()

        Mu_kk_mean = Sum.sum_sc_field(self.Mu_kk)

        knl_norm_sqr_field_mat(self.LB, self.mB, self.LbmB_field)

        LbmB = Sum.sum_sc_field(self.LbmB_field)

        knl_norm_sqr_field_mat(self.mB, self.mB, self.mBmB_field)

        mBmB = Sum.sum_sc_field(self.mBmB_field)

        # print(LuM, Lkk_mean, LbmB)
        # print(MM_mean, Mu_kk_mean, mBmB)
        self.C = LuM / MM_mean
        self.Y = Lkk_mean / Mu_kk_mean
        self.D = LbmB / mBmB

    def update_les(self):
        if self.les != NonHallLES.DNS:
            self.get_les_consts()

    @ti.kernel
    def get_cfl_cond_les(self) -> double:
        max_nu = double(0.0)
        max_eta = double(0.0)
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                nu = 2.0*ti.abs(self.get_alpha_ij(idx)*self.C + self.Y*self.get_tr_alpha(idx)) / self.Q_rho(idx)
                eta = 2.0*ti.abs(self.get_phi(idx)*self.D)
                ti.atomic_max(max_nu, nu)
                ti.atomic_max(max_eta, eta)

        return ti.abs(max_eta + max_nu)

@ti.data_oriented
class LesMomentumCompute(MomentumCompute):
    def init_data(self, C, Y):
        self.C = C
        self.Y = Y

    @ti.func
    def flux_les(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        S = 0.5*(grad_U + grad_U.transpose())
        S_abs = ti.sqrt(2)*S.norm()
        nu = self.get_nu(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)

        return (
            self.C * nu[0] * (S - (1.0/3.0) * grad_U.trace() * kron) 
            + (1.0/3.0) * self.Y * nu[1] * S_abs * kron
        )

    @ti.func
    def get_nu(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        if ti.static(self.les == NonHallLES.SMAG):
            return self.nu_t_smag(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)
        elif ti.static(self.les == NonHallLES.CROSS_HELICITY):
            return self.nu_t_crosshelicity(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)
        else:
            return self.nu_t_dns(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)

    @ti.func
    def nu_t_dns(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        return vec2(1e-6)

    @ti.func
    def nu_t_smag(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        S = 0.5*(grad_U + grad_U.transpose())
        S_abs = ti.sqrt(2)*S.norm()

        nu_0 = -2.0 * self.filter_delta.norm_sqr() * V_rho * S_abs
        nu_1 = 2.0 * self.filter_delta.norm_sqr() * V_rho * S_abs
        return vec2([nu_0, nu_1])

    @ti.func
    def nu_t_crosshelicity(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        S = 0.5*(grad_U + grad_U.transpose())
        S_b = 0.5*(grad_B + grad_B.transpose())

        f = norm_dot_mat(S, S_b)

        nu_0 = - 2 * self.filter_delta.norm_sqr() * V_rho * f
        nu_1 = 2 * self.filter_delta.norm_sqr() * V_rho * f
        return vec2([nu_0, nu_1])

    @ti.func
    def get_alpha(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        return self.get_nu(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)[0]

    @ti.func
    def get_tr_alpha(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        return self.get_nu(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)[1]


@ti.data_oriented
class LesBCompute(BCompute):
    def init_data(self, D):
        self.D = D

    @ti.func
    def flux_les(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        J = 0.5*(grad_B - grad_B.transpose())
        eta = self.get_eta(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)
        
        return self.D * eta[0] * J

    @ti.func
    def get_eta(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        if ti.static(self.les == NonHallLES.SMAG):
            return self.eta_t_smag(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)
        elif ti.static(self.les == NonHallLES.CROSS_HELICITY):
            return self.eta_t_crosshelicity(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)
        else:
            return self.eta_t_dns(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)

    @ti.func
    def eta_t_dns(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        return vec1(1e-6)

    @ti.func
    def eta_t_smag(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        j = rot_B

        eta_0 = 2.0 * self.filter_delta.norm_sqr() * j.norm()
        return vec1(eta_0)

    @ti.func
    def eta_t_crosshelicity(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        j = rot_B
        w = rot_U

        jw = j.dot(w)

        eta_0 = 2.0 * self.filter_delta.norm_sqr() * ti.math.sign(jw) * ti.sqrt(ti.abs(jw))
        return vec1(eta_0)

    @ti.func
    def get_phi(self, V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B):
        return self.get_eta(V_rho, V_u, V_b, grad_U, grad_B, rot_U, rot_B)[0]
