import taichi as ti

from taichi_src.kernels.common.diff import *

@ti.func
def _get_ghost_new_idx(ghost, size, idx):
    new_idx = idx
    if idx < ghost:
        new_idx += (size - 2*ghost)
    elif idx >= size - ghost:
        new_idx -= (size - 2*ghost)
    return new_idx

@ti.func
def get_ghost_new_idx(ghost, shape, idx):
    new_idx = idx
    
    for i in ti.static(range(3)):
        new_idx[i] = _get_ghost_new_idx(ghost, shape[i], idx[i])

    return new_idx

@ti.data_oriented
class Compute:
    @ti.func
    def _check_ghost(self, shape, idx):
        return (idx < self.ghost) or (idx >= shape - self.ghost)

    @ti.func
    def check_ghost_idx(self, idx):
        result = False

        for i in ti.static(range(len(idx))):
            result = result or self._check_ghost(self.shape[i], idx[i])

        return result

    def compute_call(self, out, dt):
        self.compute(out, dt)

    def ghosts_call(self, out):
        self.ghosts(out)

# ABC compute class

# RHO

@ti.data_oriented
class RhoCompute(Compute):
    def __init__(self, h):
        self.h = vec3(h)

    def init_data(self, ghost_cell_num, rho, u):
        self.u = u

        self.rho = rho
        self.shape = self.rho.shape
        self.ghost = ghost_cell_num
        
    @ti.func
    def rho_u(self, idx):
        return self.rho[idx] * self.u[idx]

    @ti.kernel
    def compute(self, out: ti.template(), dt: ti.f64):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                out[idx] = self.rho[idx] - dt*div_vec3(self.rho_u, self.h, idx)

    @ti.kernel    
    def ghosts(self, out: ti.template()):
        for idx in ti.grouped(self.rho):
            if self.check_ghost_idx(idx):
                out[idx] = out[get_ghost_new_idx(self.ghost, self.shape, idx)]

            # print("idx: ", idx, "\n    rho_old: ", self.rho[idx], " | div: ", div_vec3(self.rho_u, idx), " | result: ", out[idx])

# RHO

# P

@ti.data_oriented
class pCompute(Compute):
    def __init__(self, gamma, h):
        self.gamma = gamma
        self.h = vec3(h)
    
    def init_data(self, ghost_cell_num, rho):
        self.rho = rho
        self.shape = rho.shape
        self.ghost = ghost_cell_num
        
    @ti.kernel
    def compute(self, out: ti.template(), dt: double):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                    out[idx] = ti.math.pow(ti.cast(self.rho[idx], ti.f32), ti.cast(self.gamma, ti.f32))
                
    @ti.kernel    
    def ghosts(self, out: ti.template()):
        for idx in ti.grouped(self.rho):
            if self.check_ghost_idx(idx):
                out[idx] = out[get_ghost_new_idx(self.ghost, self.shape, idx)]

# P

# RHOU
@ti.data_oriented
class uCompute(Compute):
    def __init__(self, Re, Ma, h):
        self.Re = Re
        self.Ma = Ma
        self.h = vec3(h)

    def init_data(self, ghost, rho, p, u, B):
        self.u = u
        self.p = p
        self.B = B
        self.rho = rho

        self.ghost = ghost
        self.shape = u.shape
        
    @ti.func
    def rho_uu(self, idx):
        return self.rho[idx] * tensor_dot_vec3(self.u[idx], self.u[idx])

    @ti.func
    def p_delta(self, idx):
        return self.p[idx] * kron

    @ti.func
    def BB_delta(self, idx):
        return (self.B[idx].norm()**2 / (2*self.Ma**2) ) * kron

    @ti.func
    def BB(self, idx):
        return tensor_dot_vec3(self.B[idx], self.B[idx]) / self.Ma**2

    @ti.func
    def get_u(self, idx):
        return self.u[idx]

    @ti.func
    def diSij(self, idx):
        result = mat3x3(0)

        for i, j in ti.static(ti.ndrange(3, 3)):
            result[i, j] = 0.5 * (
                ddx(self.get_u, [j,], i, i, self.h[i], self.h[j], idx) 
                + ddx(self.get_u, [i,], i, j, self.h[i], self.h[j], idx)
                )
        return result

    @ti.func
    def sigma(self, idx):
        return (
            2 * self.diSij(idx) 
            - (2.0/3.0) * self.diSij(idx) * kron
            ) / self.Re

    @ti.func
    def flux(self, idx):
        return (
            div_mat3x3(self.rho_uu, 1, self.h, idx)
            + div_mat3x3(self.p_delta, 1, self.h, idx)
            + div_mat3x3(self.BB_delta, 1, self.h, idx)
            + div_mat3x3(self.BB, 1, self.h, idx)
        )

    @ti.func
    def diff(self, idx):
        result = vec3(0)
        sigma = self.sigma(idx)

        for i, j in ti.ndrange(3, 3):
            result[i] += sigma[i, j]

        return result

    @ti.kernel
    def compute(self, out: ti.template(), rho_new: ti.template(), dt: ti.f64):
        for idx in ti.grouped(self.u):
            if not self.check_ghost_idx(idx):
                out[idx] = (self.rho[idx]*self.u[idx] 
                    - dt*self.flux(idx) 
                    + dt*self.diff(idx)
                ) / rho_new[idx]
            
            

    @ti.kernel    
    def ghosts(self, out: ti.template()):
        for idx in ti.grouped(self.u):
            if self.check_ghost_idx(idx):
                out[idx] = out[get_ghost_new_idx(self.ghost, self.shape, idx)]

    def compute_call(self, out, rho_new, dt):
        self.compute(out, rho_new, dt)

# RHOU

# B
@ti.data_oriented
class BCompute(Compute):
    def __init__(self, Rem, h):
        self.Rem = Rem
        self.h = vec3(h)

    def init_data(self, ghost, u, B):
        self.u = u
        self.B = B

        self.ghost = ghost
        self.shape = B.shape

    @ti.func
    def uB(self, idx):
        return tensor_dot_vec3(self.u[idx], self.B[idx])

    @ti.func
    def flux(self, idx):
        return div_mat3x3(self.uB, 0, self.h, idx) - div_mat3x3(self.uB, 1, self.h, idx)

    @ti.func
    def get_B(self, idx):
        return self.B[idx]

    @ti.func
    def diff(self, idx):
        result = vec3(0)

        for i, j in ti.static(ti.ndrange(3, 3)):
            result[i] += ddx(self.get_B, [i, ], j, j, self.h[j], self.h[j], idx)
        return result / self.Rem

    @ti.kernel
    def compute(self, out: ti.template(), dt: ti.f64):
        for idx in ti.grouped(self.B):
            if not self.check_ghost_idx(idx):
                out[idx] = (
                    self.B[idx] 
                    - dt*self.flux(idx)
                    + dt*self.diff(idx)
                )

    @ti.kernel    
    def ghosts(self, out: ti.template()):
        for idx in ti.grouped(self.B):
            if self.check_ghost_idx(idx):
                out[idx] = out[get_ghost_new_idx(self.ghost, self.shape, idx)]

# B