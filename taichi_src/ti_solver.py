import numpy as np
import taichi as ti

from common.logger import Logger
from common.data_service import DataService

from taichi_src.common.types import *
from taichi_src.spatial_computer.les_computers_fv import LesComputer

@ti.data_oriented
class TiSolver:
    def __init__(self, config, data_path='', arch=ti.cpu):
        # self.C = 0
        # self.Y = 0
        # self.D = 0

        self.RHO0 = 1e0
        self.U0 = 1e0
        self.B0 = 1e0
        self.eps_p = 1e-5
        self.h = [0, 0, 0]
        self.Re = 1e3
        self.Rem = 1e3
        self.delta_hall = 1e1
        self.Ma = 1e0
        self.Ms = 1e0
        self.gamma = 7.0/5.0

        self.CFL = 0.9

        self.rk_steps = 3

        self.config = config
        self.ghost = self.config.ghosts
        self.shape = self.config.shape
        for i, l in enumerate(self.config.domain_size):
            self.h[i] = l / self.config.true_shape[i]

        self.data_service = DataService(dir_name=data_path,rw_energy=self.config.rewrite_energy)

        ti.init(arch=arch, debug=True)

        self.u = [ti.Vector.field(n=3, dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]
        self.B = [ti.Vector.field(n=3, dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]
        self.p = [ti.field(dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]
        self.rho = [ti.field(dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]

        self.fv_computer = LesComputer(self.gamma, self.Re, self.Ma, self.Ms, self.Rem, 
            self.delta_hall, self.ghost, self.config.shape, self.h, self.config.domain_size, ideal=False, hall=True,
            #  les=NonHallLES.DNS)
             les=NonHallLES.SMAG)

    def read_file(self, i):
        self.current_time, rho_, p_, u_, B_ = self.data_service.read_data(i)

        self.rho[0].from_numpy(rho_)
        self.p[0].from_numpy(p_)
        self.u[0].from_numpy(u_)
        self.B[0].from_numpy(B_)

    def save_file(self, i):
        Logger.log(f'Writing step_{self.current_step}, t: {self.current_time} to file.')
        self.data_service.save_data(i, (self.current_time, self.rho[0].to_numpy(), 
            self.p[0].to_numpy(), self.u[0].to_numpy(), self.B[0].to_numpy()))
        Logger.log(f'Writind step_{self.current_step}, t: {self.current_time} to file - done!')

    def solve(self):
        self.current_time = 0
        self.current_step = self.config.start_step

        if self.config.start_step == 0:
            Logger.log('Start solve initials.')
            # self.initials_rand()
            self.initials_OT()
            self.initials_ghosts()
            self.save_file(self.current_step)
            Logger.log('Initials - done!')
        else:
            Logger.log(f'Start solve from step: {self.config.start_step}.')
            self.read_file(self.config.start_step)
            Logger.log(f'Start solve time: {self.current_time}, end_time: {self.config.end_time}.')

        Logger.log('Start solving.')

        while self.current_time < self.config.end_time or (self.current_step % self.config.rw_del != 0):

            # dT = 1e-1 * ( min(self.config.domain_size) / max(self.config.true_shape))**2
            dT = self.get_cfl()

            self.current_time += dT
            self.current_step += 1

            self.FV_step(dT)

            if self.config.model.lower() != 'dns':
                self.compute_coefs()

            if self.current_step % self.config.rw_del == 0:
                self.save_file(self.current_step)

    def get_cfl(self):
        self.fv_computer.update_data(self.rho[0], self.p[0], self.u[0], self.B[0])
        lambdas = self.fv_computer.get_cfl_cond()

        return self.CFL / (
            (lambdas[0] / self.h[0] + lambdas[1] / self.h[1] + lambdas[2] / self.h[2]) 
            + 2 * (1/self.h[0]**2 + 1/self.h[1]**2 / 1/self.h[2]**2)
        )
        # (1/self.Re + 1/self.Rem)

    @ti.func
    def _check_ghost(self, shape, idx):
        return (idx < self.config.ghosts) or (idx >= shape - self.config.ghosts)

    @ti.func
    def check_ghost_idx(self, shape, idx):
        result = False

        for i in ti.static(range(len(idx))):
            result = result or self._check_ghost(shape[i], idx[i])

        return result

    @ti.kernel
    def initials_OT(self):
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            self.rho[0][idx] = self.RHO0
            self.p[0][idx] = pow(ti.cast(self.RHO0, ti.f32), ti.cast(self.gamma, ti.f32))
            self.u[0][idx] = vec3([
                -(1 + self.eps_p*ti.math.sin(self.h[2]*z))*self.U0*ti.math.sin(self.h[1]*y),
                (1 + self.eps_p*ti.math.sin(self.h[2]*z))*self.U0*ti.math.sin(self.h[0]*x),
                self.eps_p*ti.math.sin(self.h[2]*z)
                ])
            self.B[0][idx] = vec3([
                -self.B0*ti.math.sin(self.h[1]*y),
                self.B0*ti.math.sin(2.0*self.h[0]*x),
                0
                ])

    @ti.kernel
    def initials_rand(self):
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx
            
            _rho = self.RHO0*(1 + 1e-2*ti.randn(dt=double))
            self.rho[0][idx] = _rho
            self.p[0][idx] = pow(ti.cast(_rho, ti.f32), ti.cast(self.gamma, ti.f32))

            self.u[0][idx] = self.U0 * (vec3(1) + 1e-2*vec3([ti.randn(dt=double), ti.randn(dt=double), ti.randn(dt=double)]))
            self.B[0][idx] = self.B0 * (vec3(1) + 1e-2*vec3([ti.randn(dt=double), ti.randn(dt=double), ti.randn(dt=double)]))

    def initials_ghosts(self):
        self.fv_computer.ghosts_call(self.rho[0])
        self.fv_computer.ghosts_call(self.p[0])
        self.fv_computer.ghosts_call(self.u[0])
        self.fv_computer.ghosts_call(self.B[0])

    @ti.kernel
    def sum_fields(self, a: ti.template(), b:ti.template(), c:ti.template(), c1:double, c2:double, c3:double):
        for idx in ti.grouped(a):
              c[idx] = c1*a[idx] + c2*b[idx] + c3*c[idx]

    @ti.kernel
    def sum_fields_u(self, a: ti.template(), b:ti.template(), c:ti.template(), rho_old: ti.template(), rho_new:ti.template(), c1:double, c2:double, c3:double):
        for idx in ti.grouped(a):
            c[idx] = (c1*a[idx]*rho_old[idx] + c2* b[idx]) / (rho_new[idx]) + c3*c[idx]
    
    def FV_step(self, dT):
        coefs = [(1, 0.5 * dT, 0), (1, 0.5 * dT, 0), (2.0/3.0, (2.0/3.0) * dT, 1.0/3.0)]

        self.fv_computer.update_les()
        for i, c in enumerate(coefs):
            
            i_next = (i + 1) % self.rk_steps

            i_k = i_next
            if i_k == 0:
                i_k += 1

            self.fv_computer.update_data(self.rho[i], self.p[i], self.u[i], self.B[i])

            self.fv_computer.computeRho(self.rho[i_k])
            self.fv_computer.ghosts_call(self.rho[i_k])

            self.fv_computer.computeB(self.B[i_k])
            self.fv_computer.ghosts_call(self.B[i_k])

            self.fv_computer.computeRHO_U(self.u[i_k])
            self.fv_computer.ghosts_call(self.u[i_k])

            self.sum_fields(self.rho[i], self.rho[i_k], self.rho[i_next], c[0], c[1], c[2])
            self.sum_fields_u(self.u[i], self.u[i_k], self.u[i_next], self.rho[i], self.rho[i_next], c[0], c[1], c[2])
            self.sum_fields(self.B[i], self.B[i_k], self.B[i_next], c[0], c[1], c[2])

            self.fv_computer.computeP(self.p[i_next], self.rho[i_next])
            self.fv_computer.ghosts_call(self.p[i_next])