class LesComputer(SystemComputer):
    def _get_new_shape(self, domain_size, old_h, filter_size):
        new_h = old_h * filter_size
        new_shape = int(domain_size // new_h)
        return new_shape, domain_size / new_shape

    def filter_scalar(self, foo, filter_size):
        new_shape = [0, 0, 0]
        new_h = self.h
        for i in range(len(new_h)):
            new_shape[i], new_h[i] = self._get_new_shape(self.domain_size[i], self.h[i], filter_size[i])

        new_field = ti.field(dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        self.foo_filter(foo, new_field, self.shape, new_h, self.h)

        return new_field

    def filter_scalar_favre(self, foo, filter_size):
        new_shape = [0, 0, 0]
        new_h = self.h
        for i in range(len(new_h)):
            new_shape[i], new_h[i] = self._get_new_shape(self.domain_size[i], self.h[i], filter_size[i])

        new_field = ti.field(dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        self.foo_filter_favre(foo, new_field, self.shape, new_h, self.h)
        return new_field

    def filter_vec(self, foo, filter_size):
        new_shape = [0, 0, 0]
        new_h = self.h
        for i in range(len(new_h)):
            new_shape[i], new_h[i] = self._get_new_shape(self.domain_size[i], self.h[i], filter_size[i])

        new_field = ti.Vector.field(n=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)

        self.foo_filter(foo, new_field, self.shape, new_h, self.h)
        return new_field

    def filter_vec_favre(self, foo, filter_size):
        new_shape = [0, 0, 0]
        new_h = self.h
        for i in range(len(new_h)):
            new_shape[i], new_h[i] = self._get_new_shape(self.domain_size[i], self.h[i], filter_size[i])

        new_field = ti.Vector.field(n=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)

        self.foo_filter_favre(foo, new_field, self.shape, new_h, self.h)
        return new_field

    def filter_mat(self, foo, filter_size):
        new_shape = [0, 0, 0]
        new_h = self.h
        for i in range(len(new_h)):
            new_shape[i], new_h[i] = self._get_new_shape(self.domain_size[i], self.h[i], filter_size[i])

        new_field = ti.Matrix.field(n=3, m=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)

        self.foo_filter(foo, new_field, self.shape, new_h, self.h)
        return new_field

    def filter_mat_favre(self, foo, filter_size):
        new_shape = [0, 0, 0]
        new_h = self.h
        for i in range(len(new_h)):
            new_shape[i], new_h[i] = self._get_new_shape(self.domain_size[i], self.h[i], filter_size[i])

        new_field = ti.Matrix.field(n=3, m=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)

        self.foo_filter_favre(foo, new_field, self.shape, new_h, self.h)
        return new_field

    @ti.kernel
    def foo_filter(self, foo: ti.template(), out: ti.template(), 
        old_shape: vec3i, new_h: vec3, h: vec3):
        for i, j, k in ti.ndrange((old_shape[0], old_shape[1], old_shape[2])):
            i_left_new = ti.floor(i * h[0] / new_h[0])
            i_right_new = ti.floor((i+1) * h[0] / new_h[0])

            if i_left_new == i_right_new:
                out[i_left_new] += h[0] * foo(i, j, k)
            else:
                left_delta = i_right_new*new_h[0] - i*h[0]
                out[i_left_new] += left_delta * foo(i, j, k)
                out[i_right_new] += (h[0] - left_delta) * foo(i, j, k)

            j_left_new = ti.floor(j * h[1] / new_h[1])
            j_right_new = ti.floor((j+1) * h[1] / new_h[1])

            if j_left_new == j_right_new:
                out[j_right_new] += h[1] * foo(i, j, k)
            else:
                left_delta = j_right_new*new_h[1] - i*h[1]
                out[j_left_new] += left_delta * foo(i, j, k)
                out[j_right_new] += (h[1] - left_delta) * foo(i, j, k)

            k_left_new = ti.floor(k * h[2] / new_h[2])
            k_right_new = ti.floor((k+1) * h[2] / new_h[2])

            if k_left_new == k_right_new:
                out[k_left_new] += h[2] * foo(i, j, k)
            else:
                left_delta = k_right_new*new_h[2] - k*h[2]
                out[k_left_new] += left_delta * foo(i, j, k)
                out[k_right_new] += (h[2] - left_delta) * foo(i, j, k)

    @ti.kernel
    def foo_filter_favre(self, foo: ti.template(), out: ti.template(), 
        old_shape: vec3i, new_h: vec3, h: vec3):
        for i, j, k in ti.ndrange((old_shape[0], old_shape[1], old_shape[2])):
            i_left_new = ti.floor(i * h[0] / new_h[0])
            i_right_new = ti.floor((i+1) * h[0] / new_h[0])

            if i_left_new == i_right_new:
                out[i_left_new] += h[0] * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[i_left_new]
            else:
                left_delta = i_right_new*new_h[0] - i*h[0]
                out[i_left_new] += left_delta * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[i_left_new]
                out[i_right_new] += (h[0] - left_delta) * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[i_right_new]

            j_left_new = ti.floor(j * h[1] / new_h[1])
            j_right_new = ti.floor((j+1) * h[1] / new_h[1])

            if j_left_new == j_right_new:
                out[j_right_new] += h[1] * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[j_left_new]
            else:
                left_delta = j_right_new*new_h[1] - i*h[1]
                out[j_left_new] += left_delta * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[j_left_new]
                out[j_right_new] += (h[1] - left_delta) * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[j_right_new]

            k_left_new = ti.floor(k * h[2] / new_h[2])
            k_right_new = ti.floor((k+1) * h[2] / new_h[2])

            if k_left_new == k_right_new:
                out[k_left_new] += h[2] * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[k_left_new]
            else:
                left_delta = k_right_new*new_h[2] - k*h[2]
                out[k_left_new] += left_delta * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[k_left_new]
                out[k_right_new] += (h[2] - left_delta) * foo(i, j, k) * self.rho[i, j, k] / self.rho_hat[k_right_new]

    @ti.kernel
    def knl_get_norm_field(self, a: ti.template(), b:ti.template(), out: ti.template()):
        for idx in ti.grouped(out):
            out[idx] = norm_dot(a[idx], b[idx])

    def get_rho(self, idx):
        return self.rho[idx]

    def get_B(self, idx):
        return self.B[idx]

    def rhoU(self, idx):
        return self.rho[idx]*self.u[idx]

    def Lu_a(self, idx):
        rhoU = self.rhoU(idx)
        return rhoU.outer_product(rhoU) / self.rho[idx]

    def Lu_b(self, idx):
        B = self.B[idx]
        return B.outer_product(B)

    def Lb_a(self, idx):
        return self.rhoU(idx).outer_product(self.B[idx]) / self.rho[idx]

    @ti.kernel
    def get_Lu(self, L: ti.template(), Lu_a: ti.template(),
        Lu_b: ti.template(), rho_hat: ti.template(), 
        rhoU_hat: ti.template(), B_hat: ti.template()):
        for idx in ti.grouped(L):
            L[idx] = ( Lu_a[idx] 
                - rhoU_hat[idx].outer_product(rhoU_hat[idx]) / rho_hat[idx]
                - (Lu_b[idx] - B_hat[idx].outer_product(B_hat[idx])) / self.Ma**2
            )

    @ti.kernel
    def get_Lb(self, L: ti.template(), Lb_a: ti.template(),
        rho_hat: ti.template(), rhoU_hat: ti.template(), B_hat: ti.template()):
        for idx in ti.grouped(L):
            res = Lb_a[idx] - rhoU_hat[idx].outer_product(B_hat[idx]) / rho_hat[idx]
            L[idx] = res - res.transpose()

    @ti.kernel
    def get_mat_foo(self, foo: ti.template(), out: ti.template()):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                gradU = grad_vec(self.V, self.h, idx)
                out[idx] = 0.5*(gradU + gradU.transpose())

    def get_mat_foo_call(self, foo):
        out = self.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        self.get_mat_foo(foo, out)
        self.ghost(out)
        return out

    @ti.func
    def get_Su(self, idx):
        gradU = grad_vec(self.V, self.h, idx)
        return 0.5*(gradU + gradU.transpose())

    @ti.func
    def get_Su_abs(self, idx):
        Su = self.get_Su(idx)
        return ti.sqrt(2)*Su.norm()

    @ti.func
    def get_J(self, idx):
        gradB = grad_vec(self.V, self.h, idx)
        return 0.5*(gradB - gradB.transpose())

    @ti.func
    def get_Mu_a(self, idx):
        Su = self.get_Su(idx)
        alpha = self.u_computer.get_nu(idx)

        return alpha[0] * (Su - (1.0/3.0) * Su.trace()*kron)

    @ti.func
    def get_Mu_kk_a(self, idx):
        Su = self.get_Su(idx)
        alpha = self.u_computer.get_nu(idx)

        return alpha[1] * ti.sqrt(2) * Su.norm()

    @ti.func
    def get_mB_a(self, idx):
        J = self.get_J(idx)
        phi = self.B_computer.get_etha(idx)

        return phi[0] * J

    @ti.func
    def get_alpha_ij(self, idx):
        return self.u_computer.get_nu(idx)[0]

    @ti.func
    def get_alpha_kk(self, idx):
        return self.u_computer.get_nu(idx)[1]

    @ti.func
    def get_phi(self, idx):
        return self.B_computer.get_etha(idx)[0]

    @ti.kernel
    def get_Mu(self, Mu: ti.template(), Mu_a: ti.template(), alpha_hat: ti.template(), Su_hat: ti.template()):
        for idx in ti.grouped(Mu):
            Mu[idx] = alpha_hat[idx] * (Su_hat[idx] - (1.0/3.0)*kron*Su_hat[idx].trace()) - Mu_a[idx]

    @ti.kernel
    def get_Mu_kk(self, Mu_kk: ti.template(), Mu_kk_a: ti.template(), alpha_kk_hat: ti.template(), Su_abs_hat: ti.template()):
        for idx in ti.grouped(Mu):
            Mu_kk[idx] = alpha_kk_hat[idx] * Su_abs_hat[idx] - Mu_kk_a[idx]

    @ti.kernel
    def get_mB(self, mB: ti.template(), mB_a: ti.template(), phi_hat: ti.template(), J_hat: ti.template()):
        for idx in ti.grouped(Mu):
            mB[idx] = phi_hat[idx] * J_hat[idx] - mB_a[idx]

    @ti.kernel
    def mat_norm_field(self, a: ti.template(), b: ti.template(), out: ti.template()):
        for idx in ti.grouped(out):
            out[idx] = norm_dot(a[idx], b[idx])

    @ti.kernel
    def mat_tr_sqr_field(self, a: ti.template(), out: ti.template()):
        for idx in ti.grouped(out):
            out[idx] = get_trace_sqr(a)

    def get_les_coefs(self):
        filter_size = vec3i(2, 2, 2)
        
        self.rho_hat = self.filter_scalar(self.get_rho, filter_size)
        self.rhoU_hat = self.filter_vec(self.rhoU, filter_size)
        self.B_hat = self.filter_vec(self.get_B, filter_size)

        Lu_a_hat = self.filter_mat(self.Lu_a, filter_size)
        Lu_b_hat = self.filter_mat(self.Lu_b, filter_size)
        Lu = ti.Matrix.field(n=3, m=3, dtype=rho_hat.dtype, shape=rho.shape)
        self.get_Lu(Lu, Lu_a_hat, Lu_b_hat, rho_hat, rhoU_hat, B_hat)

        Lb_a = self.filter_mat(self.Lb_a, filter_size)
        LB = ti.Matrix.field(n=3, m=3, dtype=rho_hat.dtype, shape=rho.shape)
        self.get_Lb(LB, Lb_a, rho_hat, rhoU_hat, B_hat)

        Su = self.get_mat_foo_call(self.get_Su)
        Mu_a = self.get_mat_foo_call(self.get_Mu_a)
        alpha_ij = self.get_mat_foo_call(self.get_alpha_ij)
        
        
        Mu_a_hat = self.filter_mat_favre(Mu_a, filter_size)
        Su_hat = self.filter_mat_favre(Su, filter_size)
        alpha_ij_hat = self.filter_scalar(alpha_ij, filter_size)

        Mu = ti.Matrix.field(n=3, m=3, dtype=rho_hat.dtype, shape=rho.shape)
        self.get_Mu(Mu, Mu_a_hat, alpha_hat, Su_hat)

        J = self.get_mat_foo_call(self.get_J)
        mB_a = self.get_mat_foo_call(self.get_mB_a)

        phi_hat = self.filter_scalar(self.get_phi, filter_size)
        
        mB_a_hat = self.filter_mat(mB_a, filter_size)
        J_hat = self.filter_mat(J, filter_size)

        mB = ti.Matrix.field(n=3, m=3, dtype=rho_hat.dtype, shape=rho.shape)
        self.get_mB(mB, mB_a_hat, phi_hat, J_hat)

        Su_abs = self.get_mat_foo_call(self.get_Su_abs)
        Mu_kk_a = self.get_mat_foo_call(self.get_Mu_kk_a)
        alpha_kk = self.get_mat_foo_call(self.get_alpha_kk)

        Su_abs_hat = self.filter_scalar_favre(Su_abs, filter_size)
        Mu_kk_a_hat = self.filter_scalar_favre(Mu_kk_a, filter_size)
        alpha_kk_hat = self.filter_scalar(alpha_kk, filter_size)

        Mu_kk = ti.field(dtype=rho_hat.dtype, shape=rho.shape)
        self.get_Mu_kk(Mu, Mu_kk_a_hat, alpha_kk_hat, Su_abs_hat)


        LuM_field = ti.field(dtype=rho_hat.dtype, shape=rho.shape)
        self.mat_norm_field(Lu, Lu, LuM_field)

        LuM = LuM_field.sum()

        MM_field = ti.field(dtype=rho_hat.dtype, shape=rho.shape)
        self.mat_norm_field(Mu, Mu, MM_field)

        MM = MM_field.sum()

        Lkk_field = ti.field(dtype=rho_hat.dtype, shape=rho.shape)
        self.mat_tr_sqr_field(Lu, Lkk_field)

        Lkk_mean = Lkk_field.sum()

        Mu_kk_mean = Mu_kk.sum()

        LbmB_field = ti.field(dtype=rho_hat.dtype, shape=rho.shape)
        self.mat_norm_field(Lb, mB, LbmB_field)

        LbmB = LbmB_field.sum()

        mBmB_field = ti.field(dtype=rho_hat.dtype, shape=rho.shape)
        self.mat_norm_field(mB, mB, mBmB_field)

        mBmB = mBmB_field.sum()

        self.C = LuM / MM
        self.Y = Lkk_mean / Mu_kk_mean
        self.D = LbmB / mBmB

    def update_data(self, rho, p, u, B):
        self.u = u
        self.p = p
        self.B = B
        self.rho = rho

        self.get_les_coefs()
        self.rho_computer.init_data(rho, u)
        self.B_computer.init_data(rho, u, B, D=self.D)
        self.u_computer.init_data(rho, p, u, B, C=self.C, Y=self.Y)