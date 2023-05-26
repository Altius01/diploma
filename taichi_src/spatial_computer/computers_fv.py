import taichi as ti
from abc import ABC, abstractmethod

from taichi_src.common.types import *
from taichi_src.common.boundaries import *
from taichi_src.common.matrix_ops import *
from taichi_src.spatial_diff.diff_fv import *

@ti.data_oriented
class SystemComputer:
    def __init__(self, gamma, Re, Ms, Ma, Rem, delta_hall, 
        ghosts, shape, h, domain_size, ideal=False, hall=False, les=NonHallLES.DNS):
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

        self.rho_computer = RhoCompute(self.h, filter_size=self.filter_size, les=les)
        self.u_computer = MomentumCompute(Re, Ma, self.h, filter_size=self.filter_size, les=les)
        self.B_computer = BCompute(Rem, delta_hall, self.h, filter_size=self.filter_size, les=les)

    def update_data(self, rho, p, u, B):
        self.u = u
        self.p = p
        self.B = B
        self.rho = rho

        self.rho_computer.init_data(rho, u)
        self.B_computer.init_data(rho, u, B)
        self.u_computer.init_data(rho, p, u, B)        

    @ti.func
    def _check_ghost(self, shape, idx):
        return (idx < self.ghost) or (idx >= shape - self.ghost)

    @ti.func
    def check_ghost_idx(self, idx):
        result = False

        for i in ti.static(range(idx.n)):
            result = result or self._check_ghost(self.shape[i], idx[i])

        return result

    @ti.func
    def get_eigenvals(self, j, idx):
        u = self.u[idx]
        B = self.B[idx]
        p = self.p[idx]
        rho = self.rho[idx]

        pi_rho = ti.sqrt(4 * ti.math.pi * rho)

        b = (1 / self.Ma) * B.norm() / pi_rho
        b_x = (1 / self.Ma) * B[j] / pi_rho
        # Sound speed
        c = (1 / self.Ms) * ti.sqrt(self.gamma * p / rho)
        # Alfen speed
        c_a = (1 / self.Ma) * B[j] / pi_rho

        sq_root = ti.sqrt((b + c)**2 - 4 * b_x * c)

        # Magnetosonic wawes
        c_s = ti.sqrt( 0.5 * (b + c) - sq_root)
        c_f = ti.sqrt( 0.5 * (b + c) + sq_root)

        return ti.Vector([
            u[j] + c_f,
            u[j] - c_f,
            u[j] + c_s,
            u[j] - c_s,
            u[j] + c_a,
            u[j] - c_a,
            u[j],
            0,
        ])

    @ti.func
    def get_s_j_max(self, j, idx):
        lambdas = ti.abs(self.get_eigenvals(j, idx))

        result = double(0.0)
        for i in ti.ndrange(lambdas.n):
            result = ti.max(result, lambdas[i])

        return result

    @ti.func
    def get_s_max(self, idx):
        result = vec3(0)

        for i in ti.static(range(3)):
            result[i] = self.get_s_j_max(i, idx)

        return result

    @ti.kernel
    def get_cfl_cond(self) -> vec3:
        result = vec3(0)
        for idx in ti.grouped(self.rho):
            result = ti.max(self.get_s_max(idx), result)

        return result

    @ti.func
    def flux_right(self, flux_conv: ti.template(), flux_viscos: ti.template(), 
        flux_hall: ti.template(), flux_les: ti.template(), Q: ti.template(), i, idx):
        idx_left = idx
        idx_right = idx + get_basis(i)

        corner_left = idx - vec3i(1) + get_basis(i)
        corner_right = idx

        s_max = self.get_s_j_max(i, idx)

        result = ( get_mat_col(flux_conv(idx_left) + flux_conv(idx_right), i)
            - s_max * (Q(idx_right) - Q(idx_left)) )

        if ti.static(self.ideal==False):
            result -= get_mat_col(flux_viscos(corner_left) + flux_viscos(corner_right), i)

        if ti.static(self.hall):
            result -= get_mat_col(flux_hall(corner_left) + flux_hall(corner_right), i)
        
        return 0.5*result

    @ti.kernel
    def computeRho(self, out: ti.template()):
        computer = self.rho_computer
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                res = double(0.0)
                for j in ti.static(range(self.h.n)):
                    idx_r = idx
                    idx_l = idx - get_basis(j)

                    flux_r = self.flux_right(computer.flux_convective, computer.flux_viscous,
                    computer.flux_hall, computer.flux_les, computer.Q, j, idx_r)
                    
                    flux_l = self.flux_right(computer.flux_convective, computer.flux_viscous,
                    computer.flux_hall, computer.flux_les, computer.Q, j, idx_l)

                    res -= (flux_r - flux_l) / self.h[j]
                out[idx] = res

    @ti.kernel
    def computeP(self, out: ti.template(), rho_new: ti.template()):
        for idx in ti.grouped(rho_new):
            if not self.check_ghost_idx(idx):
                out[idx] = ti.math.pow(ti.cast(rho_new[idx], ti.f32), ti.cast(self.gamma, ti.f32))

    @ti.kernel
    def computeRHO_U(self, out: ti.template()):
        computer = self.u_computer
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                res = vec3(0)
                for j in ti.static(range(self.h.n)):
                    idx_r = idx
                    idx_l = idx - get_basis(j)

                    flux_r = self.flux_right(computer.flux_convective, computer.flux_viscous,
                        computer.flux_hall, computer.flux_les, computer.Q, j, idx_r)
                    
                    flux_l = self.flux_right(computer.flux_convective, computer.flux_viscous,
                    computer.flux_hall, computer.flux_les, computer.Q, j, idx_l)

                    res -= (flux_r - flux_l) / self.h[j]
                out[idx] = res

    @ti.kernel
    def computeB(self, out: ti.template()):
        computer = self.B_computer
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                res = vec3(0)
                for j in ti.static(range(self.h.n)):
                    idx_r = idx
                    idx_l = idx - get_basis(j)

                    flux_r = self.flux_right(computer.flux_convective, computer.flux_viscous,
                    computer.flux_hall, computer.flux_les, computer.Q, j, idx_r)
                    
                    flux_l = self.flux_right(computer.flux_convective, computer.flux_viscous,
                    computer.flux_hall, computer.flux_les, computer.Q, j, idx_l)

                    res -= (flux_r - flux_l) / self.h[j]
                out[idx] = res

    @ti.kernel
    def compute_U(self, rho_u: ti.template(), rho: ti.template()):
        for idx in ti.grouped(rho):
            rho_u[idx] /= rho[idx]

    @ti.kernel    
    def ghosts(self, out: ti.template()):
        for idx in ti.grouped(out):
            if self.check_ghost_idx(idx):
                out[idx] = out[get_ghost_new_idx(self.ghost, self.shape, idx)]

    def ghosts_call(self, out):
        self.ghosts(out)

    @ti.kernel
    def get_foo(self, foo: ti.template(), out: ti.template()):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                out[idx] = foo(idx)

    def get_field_from_foo(self, foo, out):
        self.get_foo(foo, out)
        self.ghosts(out)
        return out

    def get_sc_field_from_foo(self, foo):
        out = ti.field(dtype=double, shape=self.shape)
        return self.get_field_from_foo(foo, out)

    def get_vec_field_from_foo(self, foo):
        out = ti.Vector.field(n=3, dtype=double, shape=self.shape)
        return self.get_field_from_foo(foo, out)

    def get_mat_field_from_foo(self, foo):
        out = ti.Matrix.field(n=3, m=3, dtype=double, shape=self.shape)
        return self.get_field_from_foo(foo, out)



@ti.data_oriented
class Compute(ABC):
    def __init__(self, h, filter_size=vec3i(0), les=NonHallLES.DNS):
        self.h = h
        self.filter_size = filter_size
        self.filter_delta = self.h * self.filter_size
        self.les = les

    @abstractmethod
    def init_data(self, *args, **kwargs):
        ...

    @ti.func
    @abstractmethod
    def Q(self, idx):
        ...

    @ti.func
    @abstractmethod
    def V(self, idx):
        ...

    @ti.func
    @abstractmethod
    def flux_convective(self, idx):
        ...

    @ti.func
    @abstractmethod
    def flux_viscous(self, idx):
        ...

    @ti.func
    @abstractmethod
    def flux_hall(self, idx):
        ...

    @ti.func
    @abstractmethod
    def flux_les(self, idx):
        ...


@ti.data_oriented
class RhoCompute(Compute):
    def init_data(self, rho, u):
        self.u = u
        self.rho = rho

    @ti.func
    def Q(self, idx):
        return self.rho[idx]

    @ti.func
    def V(self, idx):
        return self.rho[idx]

    @ti.func
    def flux_convective(self, idx):
        return self.rho[idx]*self.u[idx]

    @ti.func
    def flux_viscous(self, idx):
        return vec3(0)

    @ti.func
    def flux_hall(self, idx):
        return vec3(0)

    @ti.func
    def flux_les(self, idx):
        return vec3(0)


@ti.data_oriented
class MomentumCompute(Compute):
    def __init__(self, Re, Ma, h, filter_size=vec3i(0), les=NonHallLES.DNS):
        super().__init__(h, filter_size=filter_size, les=les)
        self.Re = Re
        self.Ma = Ma

    def init_data(self, rho, p, u, B, C=0, Y=0):
        self.u = u
        self.p = p
        self.B = B
        self.rho = rho

        self.C = C
        self.Y = Y

    @ti.func
    def Q(self, idx):
        return self.rho[idx]*self.u[idx]

    @ti.func
    def V(self, idx):
        return self.u[idx]

    @ti.func
    def flux_convective(self, idx):
        return (
            self.rho_uu(idx)
            + (self.p[idx] + (0.5/self.Ma**2) * self.B[idx].norm_sqr()) * kron
            - self.BB(idx)
        )

    @ti.func
    def flux_viscous(self, idx):
        gradU = grad_vec(self.V, self.h, idx)
        divU = gradU.trace()

        return (
            gradU + gradU.transpose() + (2.0/3.0) * divU * kron
        ) / self.Re

    @ti.func
    def flux_hall(self, idx):
        return mat3x3(0)

    @ti.func
    def flux_les(self, idx):
        return mat3x3(0)

    @ti.func
    def rho_uu(self, idx):
        u = self.u[idx]
        return self.rho[idx] * u.outer_product(u)

    @ti.func
    def BB(self, idx):
        B = self.B[idx]
        return B.outer_product(B)

    @ti.func
    def get_S(self, idx):
        gradU = grad_vec(self.V, self.h, idx)
        return 0.5*(gradU + gradU.transpose())

    @ti.func
    def get_S_abs(self, idx):
        S = self.get_S(idx)
        return ti.sqrt(2)*S.norm()


@ti.data_oriented
class BCompute(Compute):
    def __init__(self, Rem, delta_hall, h, filter_size=vec3i(0), les=NonHallLES.DNS):
        super().__init__(h, filter_size=filter_size, les=les)
        self.Rem = Rem
        self.delta_hall = delta_hall

    def init_data(self, rho, u, B, D=0):
        self.rho = rho
        self.u = u
        self.B = B
        self.D = D

    @ti.func
    def Q(self, idx):
        return self.B[idx]

    @ti.func
    def V(self, idx):
        return self.B[idx]

    @ti.func
    def flux_convective(self, idx):
        Bu = self.Bu(idx)
        return Bu - Bu.transpose()

    @ti.func
    def flux_viscous(self, idx):
        gradB = grad_vec(self.V, self.h, idx)
        return (
            gradB - gradB.transpose()
        ) / self.Rem

    @ti.func
    def flux_hall(self, idx):
        j = rot_vec(self.V, self.h, idx)
        v_h = - self.delta_hall * j / V_plus_sc(self.get_rho, idx)
        v_hB = v_h.outer_product(V_plus_vec(self.V, idx))  
        return (
            v_hB - v_hB.transpose()
        )

    @ti.func
    def flux_les(self, idx):
        return mat3x3(0)

    @ti.func
    def Bu(self, idx):
        u = self.u[idx]
        B = self.B[idx]
        return B.outer_product(u)

    @ti.func
    def get_rho(self, idx):
        return self.rho[idx]

    @ti.func
    def get_u(self, idx):
        return self.u[idx]

    @ti.func
    def get_J(self, idx):
        gradB = grad_vec(self.V, self.h, idx)
        return 0.5*(gradB - gradB.transpose())

    @ti.func
    def get_J_abs(self, idx):
        J = self.get_J(idx)
        return J.norm()