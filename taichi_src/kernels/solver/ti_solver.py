import numpy as np
import taichi as ti

from common.logger import Logger
from common.data_service import DataService

from taichi_src.kernels.solver.computers import *

@ti.data_oriented
class TiSolver:
    def __init__(self, config, data_path='', arch=ti.cpu):
        global h

        self.C = 0
        self.Y = 0
        self.D = 0

        self.U0 = 1e-1
        self.B0 = 1e-1
        self.eps_p = 1e-5
        self.h = [0, 0, 0]
        self.Re = 1e2
        self.Rem = 1e2
        self.Ma = 1.5
        self.gamma = 7.0/5.0

        self.config = config
        self.ghost = self.config.ghosts
        self.shape = self.config.shape
        for i, l in enumerate(self.config.domain_size):
            self.h[i] = l / self.config.true_shape[i]

        self.data_service = DataService(dir_name=data_path,rw_energy=self.config.rewrite_energy)

        ti.init(arch=arch, debug=True)

        self.u = [ti.Vector.field(n=3, dtype=ti.f64, shape=self.config.shape) for i in range(2)]
        self.B = [ti.Vector.field(n=3, dtype=ti.f64, shape=self.config.shape) for i in range(2)]
        self.p = [ti.field(dtype=ti.f64, shape=self.config.shape) for i in range(2)]
        self.rho = [ti.field(dtype=ti.f64, shape=self.config.shape) for i in range(2)]

        self.rho_comp = RhoCompute(self.h)
        self.p_comp = pCompute(self.gamma, self.h)
        self.u_comp = uCompute(self.Re, self.Ma, self.h)
        self.B_comp = BCompute(self.Rem, self.h)

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
            self.initials()
            self.initils_ghosts()
            self.save_file(self.current_step)
            Logger.log('Initials - done!')
        else:
            Logger.log(f'Start solve from step: {self.config.start_step}.')
            self.read_file(self.config.start_step)
            Logger.log(f'Start solve time: {self.current_time}, end_time: {self.config.end_time}.')

        Logger.log('Start solving.')

        while self.current_time < self.config.end_time or (self.current_step % self.config.rw_del != 0):
            dT = 0.1 * ( min(self.config.domain_size) / max(self.config.true_shape))**2

            self.current_time += dT
            self.current_step += 1
            self._step(dT)

            if self.config.model.lower() != 'dns':
                self.compute_coefs()

            if self.current_step % self.config.rw_del == 0:
                self.save_file(self.current_step)

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
    def initials(self):
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            self.rho[0][idx] = 1
            self.p[0][idx] = 1
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
    def initils_ghosts(self):
        for x, y, z in ti.ndrange(self.ghost, 
            self.shape[1], 
                self.shape[2]):

                idx_0 = [x, y, z]
                idx_1 = [x + self.shape[0] - self.ghost, y, z]

                self.rho[0][idx_0] = self.rho[0][get_ghost_new_idx(self.config.ghosts, 
                    self.rho[0].shape, idx_0)]
                self.p[0][idx_0] = self.p[0][get_ghost_new_idx(self.config.ghosts, 
                    self.p[0].shape, idx_0)]
                self.u[0][idx_0] = self.u[0][get_ghost_new_idx(self.config.ghosts, 
                    self.u[0].shape, idx_0)]
                self.B[0][idx_0] = self.B[0][get_ghost_new_idx(self.config.ghosts, 
                    self.B[0].shape, idx_0)]

                self.rho[0][idx_1] = self.rho[0][get_ghost_new_idx(self.config.ghosts, 
                    self.rho[0].shape, idx_1)]
                self.p[0][idx_1] = self.p[0][get_ghost_new_idx(self.config.ghosts, 
                    self.p[0].shape, idx_1)]
                self.u[0][idx_1] = self.u[0][get_ghost_new_idx(self.config.ghosts, 
                    self.u[0].shape, idx_1)]
                self.B[0][idx_1] = self.B[0][get_ghost_new_idx(self.config.ghosts, 
                    self.B[0].shape, idx_1)]

        for x, y, z in ti.ndrange(self.shape[0], 
                self.ghost, 
                self.shape[2]):

                idx_0 = [x, y, z]
                idx_1 = [x, y + self.shape[1] - self.ghost, z]

                self.rho[0][idx_0] = self.rho[0][get_ghost_new_idx(self.config.ghosts, 
                    self.rho[0].shape, idx_0)]
                self.p[0][idx_0] = self.p[0][get_ghost_new_idx(self.config.ghosts, 
                    self.p[0].shape, idx_0)]
                self.u[0][idx_0] = self.u[0][get_ghost_new_idx(self.config.ghosts, 
                    self.u[0].shape, idx_0)]
                self.B[0][idx_0] = self.B[0][get_ghost_new_idx(self.config.ghosts, 
                    self.B[0].shape, idx_0)]

                self.rho[0][idx_1] = self.rho[0][get_ghost_new_idx(self.config.ghosts, 
                    self.rho[0].shape, idx_1)]
                self.p[0][idx_1] = self.p[0][get_ghost_new_idx(self.config.ghosts, 
                    self.p[0].shape, idx_1)]
                self.u[0][idx_1] = self.u[0][get_ghost_new_idx(self.config.ghosts, 
                    self.u[0].shape, idx_1)]
                self.B[0][idx_1] = self.B[0][get_ghost_new_idx(self.config.ghosts, 
                    self.B[0].shape, idx_1)]

        for x, y, z in ti.ndrange(self.shape[0], 
                self.shape[1], self.ghost):

                idx_0 = [x, y, z]
                idx_1 = [x, y, z + self.shape[2] - self.ghost]

                self.rho[0][idx_0] = self.rho[0][get_ghost_new_idx(self.config.ghosts, 
                    self.rho[0].shape, idx_0)]
                self.p[0][idx_0] = self.p[0][get_ghost_new_idx(self.config.ghosts, 
                    self.p[0].shape, idx_0)]
                self.u[0][idx_0] = self.u[0][get_ghost_new_idx(self.config.ghosts, 
                    self.u[0].shape, idx_0)]
                self.B[0][idx_0] = self.B[0][get_ghost_new_idx(self.config.ghosts, 
                    self.B[0].shape, idx_0)]

                self.rho[0][idx_1] = self.rho[0][get_ghost_new_idx(self.config.ghosts, 
                    self.rho[0].shape, idx_1)]
                self.p[0][idx_1] = self.p[0][get_ghost_new_idx(self.config.ghosts, 
                    self.p[0].shape, idx_1)]
                self.u[0][idx_1] = self.u[0][get_ghost_new_idx(self.config.ghosts, 
                    self.u[0].shape, idx_1)]
                self.B[0][idx_1] = self.B[0][get_ghost_new_idx(self.config.ghosts, 
                    self.B[0].shape, idx_1)]
            

    def _step(self, dT):
        coefs = ((0.0, 1.0), (0.75, 0.25), ((1.0/3.0), (2.0/3.0)),)

        self.rho_comp.init_data(self.config.ghosts, self.rho[0], self.u[0])
        self.rho_comp.compute_call(self.rho[1], dT)
        self.rho_comp.ghosts_call(self.rho[1])

        self.p_comp.init_data(self.config.ghosts, self.rho[1])
        self.p_comp.compute_call(self.p[1], dT)
        self.p_comp.ghosts_call(self.p[1])

        self.B_comp.init_data(self.config.ghosts, self.u[0], self.B[0])
        self.B_comp.compute_call(self.B[1], dT)
        self.B_comp.ghosts_call(self.B[1])

        self.u_comp.init_data(self.config.ghosts, self.rho[0], self.p[0], self.u[0], self.B[0])
        self.u_comp.compute_call(self.u[1], self.rho[1], dT)
        self.u_comp.ghosts_call(self.u[1])

        self.rho[0].copy_from(self.rho[1])
        self.p[0].copy_from(self.p[1])
        self.u[0].copy_from(self.u[1])
        self.B[0].copy_from(self.B[1])
