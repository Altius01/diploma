import numpy as np
import taichi as ti

from common.logger import Logger
from common.data_service import DataService

from taichi_src.common.types import *
from taichi_src.spatial_computer.computers_fv import SystemComputer
from taichi_src.spatial_computer.les_computers_fv import LesComputer

@ti.data_oriented
class TiSolver:
    def __init__(self, config, data_path='', arch=ti.cpu):
        self.RHO0 = config.RHO0
        self.U0 = config.U0
        self.B0 = config.B0
        self.eps_p = 1e-5
        self.h = [0, 0, 0]
        self.Re = config.Re
        self.Rem = config.Rem
        self.delta_hall = config.delta_hall
        self.Ma = config.Ma
        self.Ms = config.Ms
        self.gamma = config.gamma

        self.CFL = config.CFL

        self.rk_steps = 3
        self.les_model = NonHallLES(config.model)
        self.ideal = config.ideal
        self.hall = config.hall

        Logger.log(f'Starting with:')
        Logger.log(f'   Shape: {config.true_shape}')
        Logger.log(f'   CFL: {self.CFL}, LES_Model: {self.les_model}, Ideal: {self.ideal}, Hall: {self.hall}')
        Logger.log(f'   Re: {self.Re}, Rem: {self.Rem}, delta_hall: {self.delta_hall}, Ma: {self.Ma}, Ms: {self.Ms}, gamma: {self.gamma}')

        self.debug_fv_step = True
        # self.debug_fv_step = False

        self.config = config
        self.ghost = self.config.ghosts
        self.shape = self.config.shape
        for i, l in enumerate(self.config.domain_size):
            self.h[i] = l / self.config.true_shape[i]

        self.data_service = DataService(dir_name=data_path,rw_energy=self.config.rewrite_energy)

        # ti.init(arch=arch, debug=True, device_memory_GB=8)

        self.u = [ti.Vector.field(n=3, dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]
        self.B = [ti.Vector.field(n=3, dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]
        self.p = [ti.field(dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]
        self.rho = [ti.field(dtype=double, shape=self.config.shape) for i in range(self.rk_steps)]

        self.fv_computer = LesComputer(self.gamma, self.Re, self.Ms, self.Ma, self.Rem, 
            self.delta_hall, self.ghost, self.config.shape, self.h, self.config.domain_size, ideal=self.ideal, hall=self.hall,
             les=self.les_model)

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
            # self.initials_OT()
            self.initials_SOD()
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
            # print(dT)

            self.current_time += dT
            self.current_step += 1

            self.FV_step(dT)

            # if self.config.model.lower() != 'dns':
            #     self.compute_coefs()

            if self.current_step % self.config.rw_del == 0:
                self.save_file(self.current_step)

    def get_cfl(self):
        self.fv_computer.update_data(self.rho[0], self.p[0], self.u[0], self.B[0])
        lambdas = self.fv_computer.get_cfl_cond()

        denominator = (lambdas[0] / self.h[0] + lambdas[1] / self.h[1] + lambdas[2] / self.h[2])

        if (self.ideal == False):
            denominator += (1.0/self.h[0]**2 + 1.0/self.h[1]**2 + 1.0/self.h[2]**2)
        
        return self.CFL * (1.0 / denominator)

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
    def initials_SOD(self):
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            if self.h[0]*x < ti.math.pi:
                rho_ = 1.0
            else:
                rho_ = 0.125

            self.rho[0][idx] = _rho
            self.p[0][idx] = pow(rho_, self.gamma)
            self.u[0][idx] = vec3(0)
            self.B[0][idx] = vec3(0)

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

        if (self.debug_fv_step):
            print("update_les start...")
        self.fv_computer.update_les()
        if (self.debug_fv_step):
            print(f'    Les coefs: C:{self.fv_computer.C}, Y:{self.fv_computer.Y}, D: {self.fv_computer.D}')
            print("update_les done!")
        for i, c in enumerate(coefs):
            
            i_next = (i + 1) % self.rk_steps

            i_k = i_next
            if i_k == 0:
                i_k += 1

            if (self.debug_fv_step):
                print("update_data start...")
            self.fv_computer.update_data(self.rho[i], self.p[i], self.u[i], self.B[i])

            # if (self.debug_fv_step):
            #     print("update_data done!")
            #     print("computeRho start...")
            # self.fv_computer.computeRho(self.rho[i_k])
            # self.fv_computer.ghosts_call(self.rho[i_k])
            # if (self.debug_fv_step):
            #     print("computeRho done!")
            #     print("computeB start...")
            # self.fv_computer.computeB(self.B[i_k])
            # self.fv_computer.ghosts_call(self.B[i_k])
            # if (self.debug_fv_step):
            #     print("computeB done!")
            #     print("computeRHO_U start...")
            # self.fv_computer.computeRHO_U(self.u[i_k])
            # self.fv_computer.ghosts_call(self.u[i_k])
            # if (self.debug_fv_step):
            #     print("computeRHO_U done!")
            #     print("sum_fields start...")
            if (self.debug_fv_step):
                print("update_data done!")
                print("compute HLLD start...")
            self.fv_computer.computeHLLD(self.rho[i_k], self.u[i_k], self.B[i_k])
            self.fv_computer.ghosts_call(self.rho[i_k])
            self.fv_computer.ghosts_call(self.B[i_k])
            self.fv_computer.ghosts_call(self.u[i_k])
            if (self.debug_fv_step):
                print("compute HLLD done!")
                print("sum_fields start...")

            self.sum_fields(self.rho[i], self.rho[i_k], self.rho[i_next], c[0], c[1], c[2])
            self.sum_fields_u(self.u[i], self.u[i_k], self.u[i_next], self.rho[i], self.rho[i_next], c[0], c[1], c[2])
            self.sum_fields(self.B[i], self.B[i_k], self.B[i_next], c[0], c[1], c[2])
            if (self.debug_fv_step):
                print("sum_fields done!")
                print("computeP start...")
            self.fv_computer.computeP(self.p[i_next], self.rho[i_next])
            self.fv_computer.ghosts_call(self.p[i_next])
            if (self.debug_fv_step):
                print("computeP done!")

            
