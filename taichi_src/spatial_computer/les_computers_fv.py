import taichi as ti

from taichi_src.common.pointers import *
from taichi_src.common.field_ops import *
from taichi_src.spatial_computer.computers_fv import *
    
class LesComputer(SystemComputer):
    def __init__(self, gamma, Re, Ms, Ma, Rem, delta_hall, 
        ghosts, shape, h, domain_size, ideal=False, hall=False, les=NonHallLES.DNS):
        super().__init__(gamma, Re, Ms, Ma, Rem, delta_hall, 
        ghosts, shape, h, domain_size, ideal, hall, les)

        self.D = 0
        self.C = 0
        self.Y = 0

        self.u_computer = LesMomentumCompute(Re, Ma, self.h, les=les)
        self.B_computer = LesBCompute(Rem, delta_hall, self.h, les=les)

        self.init_les_arrays()

    def update_data(self, rho, p, u, B):
        self.u = u
        self.p = p
        self.B = B
        self.rho = rho

        self.rho_computer.init_data(rho, u)
        self.B_computer.init_data(rho, u, B, D=self.D)
        self.u_computer.init_data(rho, p, u, B, C=self.C, Y=self.Y)

    def get_filtered_shape(self, filter_size: vec3i):
        new_shape = [0, 0, 0]
        for i in range(len(filter_size)):
            new_shape[i] = int(self.shape[i] // filter_size[i])
        
        new_h = vec3(0)

        for i in range(len(new_h)):
            new_h[i] = self.shape[i] * self.h[i] / new_shape[i]

        return tuple(new_shape), new_h

    @ti.func
    def foo_filter_1d(self, h, new_h, i, shape, idx):
        idx_i = get_elem_1d(idx, i)
        shape_i = get_elem_1d(shape, i)
        h_i = get_elem_1d(h, i)
        new_h_i = get_elem_1d(new_h, i)

        idx_i_left = idx_i
        idx_i_right = 0
        if (idx_i + 1) < shape_i:
            idx_i_right = idx_i + 1
        else:
            idx_i_right = idx_i

        left_i_new = ti.floor(idx_i_left * h_i / new_h_i)
        right_i_new = ti.floor(idx_i_right * h_i / new_h_i)

        left_delta = h_i
        right_delta = double(0.0)
        if left_i_new != right_i_new:
            left_delta = right_i_new*new_h_i - idx_i*h_i
            right_delta = h_i - left_delta

        return vec4([left_i_new, right_i_new, left_delta, right_delta])

    @ti.kernel
    def knl_foo_filter(self, foo: ti.template(), out: ti.template(), h: vec3, new_h: vec3):
        dV_new = new_h[0]*new_h[1]*new_h[2]
        for i, j, k in ti.ndrange(self.filter_old_shape[0], 
            self.filter_old_shape[1], self.filter_old_shape[2]):
            idx = [i, j, k]

            x_vec = self.foo_filter_1d(h, new_h, 0, self.filter_old_shape, idx)
            y_vec = self.foo_filter_1d(h, new_h, 1, self.filter_old_shape, idx)
            z_vec = self.foo_filter_1d(h, new_h, 2, self.filter_old_shape, idx)

            for i, j ,k in ti.static(ti.ndrange(2, 2, 2)):
                idx_new = [0, 0, 0]
                dV = double(1.0)
                idx_new[0] = ti.cast(x_vec[i], int)
                dV *= x_vec[i+2]
                idx_new[1] =ti.cast(y_vec[j], int)
                dV *= y_vec[j+2]
                idx_new[2] = ti.cast(z_vec[k], int)
                dV *= z_vec[k+2]
                
                out[idx_new] += (foo(idx) * dV )/ dV_new

    def foo_filter(self, foo, out, shape, new_h, h):
        self.filter_old_shape = shape
        self.knl_foo_filter(foo, out, h, new_h)

    def create_filtered_sc(self, filter_size):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        new_field = ti.field(dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        return new_field

    def filter_sc(self, foo, filter_size, out):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        self.foo_filter(foo, out, self.shape, new_h, self.h)
    
    def create_filtered_vec(self, filter_size):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        new_field = ti.Vector.field(n=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        return new_field

    def filter_vec(self, foo, filter_size, out):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        self.foo_filter(foo, out, self.shape, new_h, self.h)
    
    def create_filtered_mat(self, filter_size):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        new_field = ti.Matrix.field(n=3, m=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        return new_field

    def filter_mat(self, foo, filter_size, out):
        new_shape, new_h = self.get_filtered_shape(filter_size)
        self.foo_filter(foo, out, self.shape, new_h, self.h)

    def filter_favre_sc(self, foo, rho_filtered, filter_size, out):
        self.filter_sc(foo, filter_size, out)

        field_div(out, rho_filtered)

    def filter_favre_vec(self, foo, rho_filtered, filter_size, out):
        self.filter_vec(foo, filter_size, out)

        field_div(out, rho_filtered)
    
    def filter_favre_mat(self, foo, rho_filtered, filter_size, out):
        self.filter_mat(foo, filter_size, out)

        field_div(out, rho_filtered)

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
    def get_Mu_a(self, idx):
        return self.rho[idx]*self.Mu_a[idx]

    @ti.func
    def get_alpha_ij(self, idx):
        return self.alpha_ij[idx]
    
    @ti.func
    def get_phi(self, idx):
        return self.phi[idx]

    @ti.func
    def get_mB_a(self, idx):
        return self.mB_a[idx]

    @ti.func
    def get_J(self, idx):
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
    
    def init_les_arrays(self):
        self.les_consts_filter_size = vec3i([2, 2, 2])
        filter_size = self.les_consts_filter_size
        self.rho_hat = self.create_filtered_sc(filter_size)
        self.rhoU_hat = self.create_filtered_vec(filter_size)
        self.B_hat = self.create_filtered_vec(filter_size)

        self.Lu_a_hat = self.create_filtered_mat(filter_size)
        self.Lu_b_hat = self.create_filtered_mat(filter_size)
        
        
        self.Lu = self.create_filtered_mat(filter_size)

        self.Lb_a = self.create_filtered_mat(filter_size)
        self.LB = self.create_filtered_mat(filter_size)

        self.Su = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.Mu_a = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.alpha_ij = ti.field(dtype=double, shape=self.shape)
        
        self.Mu_a_hat = self.create_filtered_mat(filter_size)
        self.Su_hat = self.create_filtered_mat(filter_size)
        self.alpha_ij_hat = self.create_filtered_sc(filter_size)

        self.Mu = self.create_filtered_mat(filter_size)

        self.J = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.mB_a = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.phi = ti.field(dtype=double, shape=self.shape)

        self.phi_hat =  self.create_filtered_sc(filter_size)
        self.mB_a_hat = self.create_filtered_mat(filter_size)
        self.J_hat = self.create_filtered_mat(filter_size)

        self.mB = self.create_filtered_mat(filter_size)

        self.Su_abs = ti.field(dtype=double, shape=self.shape)
        self.Mu_kk_a = ti.field(dtype=double, shape=self.shape)
        self.alpha_kk = ti.field(dtype=double, shape=self.shape)

        self.Su_abs_hat = self.create_filtered_sc(filter_size)
        self.Mu_kk_a_hat = self.create_filtered_sc(filter_size)
        self.alpha_kk_hat = self.create_filtered_sc(filter_size)

        self.Mu_kk = self.create_filtered_sc(filter_size)

        self.LuM_field = self.create_filtered_sc(filter_size)

        self.MM_field = self.create_filtered_sc(filter_size)

        self.Lkk_field = self.create_filtered_sc(filter_size)

        self.LbmB_field = self.create_filtered_sc(filter_size)

        self.mBmB_field = self.create_filtered_sc(filter_size)


    def get_les_consts(self):
        filter_size = self.les_consts_filter_size
        
        self.filter_sc(self.rho_computer.Q, filter_size=filter_size, out=self.rho_hat)
        self.filter_vec(self.u_computer.Q, filter_size=filter_size, out=self.rhoU_hat)
        self.filter_vec(self.B_computer.Q, filter_size=filter_size, out=self.B_hat)

        self.filter_mat(self.u_computer.Lu_a, filter_size=filter_size, out=self.Lu_a_hat)
        self.filter_mat(self.u_computer.Lu_b, filter_size=filter_size, out=self.Lu_b_hat)
    
        self.get_Lu(self.Lu, self.Lu_a_hat, self.Lu_b_hat, self.rho_hat, self.rhoU_hat, self.B_hat)
        
        self.filter_mat(self.B_computer.Lb_a, filter_size=filter_size, out=self.Lb_a)
        self.get_Lb(self.LB, self.Lb_a, self.rho_hat, self.rhoU_hat, self.B_hat)

        self.get_mat_field_from_foo(self.u_computer.get_S, out=self.Su)
        self.get_mat_field_from_foo(self.u_computer.get_Mu_a, out=self.Mu_a)
        self.get_sc_field_from_foo(self.u_computer.get_alpha, out=self.alpha_ij)
        
        self.filter_favre_mat(self.get_Mu_a, self.rho_hat, filter_size=filter_size, out=self.Mu_a_hat)
        self.filter_favre_mat(self.get_Su, self.rho_hat, filter_size=filter_size, out=self.Su_hat)
        self.filter_sc(self.get_alpha_ij, filter_size=filter_size, out=self.alpha_ij_hat)
        
        self.get_Mu(self.Mu, self.Mu_a_hat, self.alpha_ij_hat, self.Su_hat)

        self.get_mat_field_from_foo(self.B_computer.get_J, out=self.J)
        self.get_mat_field_from_foo(self.B_computer.get_mB_a, out=self.mB_a)
        self.get_sc_field_from_foo(self.B_computer.get_phi, out=self.phi)

        self.filter_sc(self.get_phi, filter_size=filter_size, out=self.phi_hat)
        self.filter_mat(self.get_mB_a, filter_size=filter_size, out=self.mB_a_hat)
        self.filter_mat(self.get_J, filter_size=filter_size, out=self.J_hat)

        self.get_mB(self.mB, self.mB_a_hat, self.phi_hat, self.J_hat)

        self.get_sc_field_from_foo(self.u_computer.get_S_abs, out=self.Su_abs)
        self.get_sc_field_from_foo(self.u_computer.get_tr_Mu_a, out=self.Mu_kk_a)
        self.get_sc_field_from_foo(self.u_computer.get_tr_alpha, out=self.alpha_kk)

        self.filter_favre_sc(self.get_Su_abs, self.rho_hat, filter_size=filter_size, out=self.Su_abs_hat)
        self.filter_favre_sc(self.get_Mu_kk_a, self.rho_hat, filter_size=filter_size, out=self.Mu_kk_a_hat)
        self.filter_sc(self.get_alpha_kk, filter_size=filter_size, out=self.alpha_kk_hat)

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

@ti.data_oriented
class LesMomentumCompute(MomentumCompute):
    @ti.func
    def flux_les(self, V_rho, V_u, V_B, idx):
        gradU = grad_vec(V_u, self.h, idx)
        S = 0.5*(gradU + gradU.transpose())

        nu = self.get_nu(idx)
        return (
            self.C * nu[0] * (S - (1.0/3.0) * S.trace() * kron) 
            + (1.0/3.0) * self.Y * nu[1] * self.get_S_abs(V_rho, V_u, V_B, idx) * kron
        )

    @ti.func
    def get_nu(self, V_rho, V_u, V_B, idx):
        if ti.static(self.les == NonHallLES.SMAG):
            return self.nu_t_smag(V_rho, V_u, V_B, idx)
        elif ti.static(self.les == NonHallLES.CROSS_HELICITY):
            return self.nu_t_crosshelicity(V_rho, V_u, V_B, idx)
        else:
            return self.nu_t_dns(V_rho, V_u, V_B, idx)

    @ti.func
    def nu_t_dns(self, V_rho, V_u, V_B, idx):
        return vec2(1e-6)

    @ti.func
    def nu_t_smag(self, V_rho, V_u, V_B, idx):
        gradU = grad_vec(V_u, self.h, idx)

        nu_0 = - 2 * self.filter_delta.norm_sqr() * V_rho(idx) * self.get_S_abs(idx)
        nu_1 = 2 * self.filter_delta.norm_sqr() * V_rho(idx) * self.get_S_abs(idx)
        return vec2([nu_0, nu_1])

    @ti.func
    def nu_t_crosshelicity(self, V_rho, V_u, V_B, idx):
        gradU = grad_vec(V_u, self.h, idx)
        gradB = grad_vec(V_B, self.h, idx)

        S = 0.5*(gradU + gradU.transpose())
        S_b = 0.5*(gradB + gradB.transpose())

        f = norm_dot_mat(S, S_b)

        nu_0 = - 2 * self.filter_delta.norm_sqr() * V_rho(idx) * f
        nu_1 = 2 * self.filter_delta.norm_sqr() * V_rho(idx) * f
        return vec2([nu_0, nu_1])

    # @ti.func
    # def get_B(self, idx):
    #     return self.B[idx]

    @ti.func
    def get_alpha(self, V_rho, V_u, V_B, idx):
        return self.get_nu(V_rho, V_u, V_B, idx)[0]

    @ti.func
    def get_tr_alpha(self, V_rho, V_u, V_B, idx):
        return self.get_nu(V_rho, V_u, V_B, idx)[1]
    
    @ti.func
    def get_Mu_a(self, V_rho, V_u, V_B, idx):
        S = self.get_S(V_rho, V_u, V_B, idx)
        alpha = self.get_nu(V_rho, V_u, V_B, idx)

        return alpha[0] * (S - (1.0/3.0) * S.trace()*kron)

    @ti.func
    def get_tr_Mu_a(self, V_rho, V_u, V_B, idx):
        alpha = self.get_nu(V_rho, V_u, V_B, idx)

        return alpha[1] * self.get_S_abs(V_rho, V_u, V_B, idx)

    @ti.func
    def Lu_a(self, V_rho, V_u, V_B, idx):
        U = V_u(idx)
        return V_rho(idx) * U.outer_product(U)

    @ti.func
    def Lu_b(self, V_rho, V_u, V_B, idx):
        B = V_B(idx)
        return B.outer_product(B)

@ti.data_oriented
class LesBCompute(BCompute):
    @ti.func
    def flux_les(self, V_rho, V_u, V_B, idx):
        gradB = grad_vec(V_B, self.h, idx)
        J = 0.5*(gradB - gradB.transpose())

        eta = self.get_eta(idx)
        
        return self.D * eta[0] * J

    @ti.func
    def get_eta(self, V_rho, V_u, V_B, idx):
        if ti.static(self.les == NonHallLES.SMAG):
            return self.eta_t_smag(idx)
        elif ti.static(self.les == NonHallLES.CROSS_HELICITY):
            return self.eta_t_crosshelicity(idx)
        else:
            return self.eta_t_dns(idx)

    @ti.func
    def eta_t_dns(self, V_rho, V_u, V_B, idx):
        return vec1(1e-6)

    @ti.func
    def eta_t_smag(self, V_rho, V_u, V_B, idx):
        j = rot_vec(V_B, self.h, idx)

        eta_0 = - 2.0 * self.filter_delta.norm_sqr() * j.norm()
        return vec1(eta_0)

    @ti.func
    def eta_t_crosshelicity(self, V_rho, V_u, V_B, idx):
        j = rot_vec(V_B, self.h, idx)
        w = rot_vec(V_u, self.h, idx)

        jw = j.dot(w)

        eta_0 = - 2 * self.filter_delta.norm_sqr() * ti.math.sign(jw) * ti.sqrt(ti.abs(jw))
        return vec1(eta_0)

    @ti.func
    def get_phi(self, V_rho, V_u, V_B, idx):
        return self.get_eta(V_rho, V_u, V_B, idx)[0]

    @ti.func
    def get_mB_a(self, V_rho, V_u, V_B, idx):
        J = self.get_J(V_rho, V_u, V_B, idx)
        phi = self.get_eta(V_rho, V_u, V_B, idx)

        return phi[0] * J

    @ti.func
    def rhoU(self, V_rho, V_u, V_B, idx):
        return V_rho(idx) * V_u(idx)

    @ti.func
    def Lb_a(self, V_rho, V_u, V_B, idx):
        return V_u(idx).outer_product(V_B(idx))
