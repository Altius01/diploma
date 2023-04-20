import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from config import Config
from logger import Logger
from cl_builder import CLBuilder
from data_service import DataService

class MHD_Solver:
    def __init__(self, context, config, data_path=''):
        self.config = config

        self.context = context
        self.queue = cl.CommandQueue(self.context)
        print(self.context)
        self.program = CLBuilder.build(self.context, self.config.defines)

        self._init_device_data()
        self._define_kernels()

        self.data_service = DataService(dir_name=data_path,rw_energy=self.config.rewrite_energy)

    def _define_kernels(self):
        self.knl_kin_e = self.program.kin_energy
        self.knl_mag_e = self.program.mag_energy
        self.knl_solve = self.program.solver_3D_RK
        self.knl_ghosts = self.program.ghost_nodes_periodic
        self.knl_initial = self.program.Orszag_Tang_3D_inital

    def _init_device_data(self):
        self.p_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.shape).astype(np.float64))
        self.u_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.v_shape).astype(np.float64))
        self.B_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.v_shape).astype(np.float64))
        self.rho_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.shape).astype(np.float64))

        self.pk1_gpu = cla.empty(self.queue, self.p_gpu.shape, dtype=np.float64)
        self.rk1_gpu = cla.empty(self.queue, self.rho_gpu.shape, dtype=np.float64)
        self.uk1_gpu = cla.empty(self.queue, self.u_gpu.shape, dtype=np.float64)
        self.Bk1_gpu = cla.empty(self.queue, self.B_gpu.shape, dtype=np.float64)

        self.pk2_gpu = cla.empty(self.queue, self.p_gpu.shape, dtype=np.float64)
        self.rk2_gpu = cla.empty(self.queue, self.rho_gpu.shape, dtype=np.float64)
        self.uk2_gpu = cla.empty(self.queue, self.u_gpu.shape, dtype=np.float64)
        self.Bk2_gpu = cla.empty(self.queue, self.B_gpu.shape, dtype=np.float64)

    def read_file(self, i):
        self.current_time, rho_, p_, u_, B_ = self.data_service.read_data(i)

        cl.enqueue_copy(self.queue, self.rho_gpu.data, rho_[:])
        cl.enqueue_copy(self.queue, self.p_gpu.data, p_[:])
        cl.enqueue_copy(self.queue, self.u_gpu.data, u_[:])
        cl.enqueue_copy(self.queue, self.B_gpu.data, B_[:])

    def save_file(self, i):
        Logger.log(f'Writing step_{self.current_step}, t: {self.current_time} to file.')
        self.data_service.save_data(i, (self.current_time, self.rho_gpu, self.p_gpu, self.u_gpu, self.B_gpu))
        Logger.log(f'Writind step_{self.current_step}, t: {self.current_time} to file - done!')

    def solve(self):
        self.current_time = 0
        self.current_step = self.config.start_step

        if self.config.start_step == 0:
            Logger.log('Start solve initials.')
            self.initials()
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

            if self.current_step % self.config.rw_del == 0:
                self.save_file(self.current_step)
                
        # if self.current_step % self.config.rw_del != 0:
        #         self.save_file(self.current_step)

    def _step(self, dT):
        _p = [self.p_gpu, self.pk1_gpu, self.pk2_gpu]
        _u = [self.u_gpu, self.uk1_gpu, self.uk2_gpu]
        _B = [self.B_gpu, self.Bk1_gpu, self.Bk2_gpu]
        _rho = [self.rho_gpu, self.rk1_gpu, self.rk2_gpu]

        coefs = ((0.0, 1.0), (0.75, 0.25), ((1.0/3.0), (2.0/3.0)),)

        evt = self.knl_solve(self.queue, self.config.true_shape, None,
                            np.float64(dT), np.float64(coefs[0][0]), np.float64(coefs[0][1]),
                            _rho[2].data, _p[2].data, _u[2].data, _B[2].data,
                            _rho[0].data, _p[0].data, _u[0].data, _B[0].data,
                            _rho[1].data, _p[1].data, _u[1].data, _B[1].data,
                            )
        evt.wait()

        self._ghost_points(_rho[1], _p[1], _u[1], _B[1])

        evt = self.knl_solve(self.queue, self.config.true_shape, None,
                            np.float64(dT), np.float64(coefs[1][0]), np.float64(coefs[1][1]),
                            _rho[0].data, _p[0].data, _u[0].data, _B[0].data,
                            _rho[1].data, _p[1].data, _u[1].data, _B[1].data,
                            _rho[2].data, _p[2].data, _u[2].data, _B[2].data,
                            )
        evt.wait()

        self._ghost_points(_rho[2], _p[2], _u[2], _B[2])

        evt = self.knl_solve(self.queue, self.config.true_shape, None,
                            np.float64(dT), np.float64(coefs[2][0]), np.float64(coefs[2][1]),
                            _rho[0].data, _p[0].data, _u[0].data, _B[0].data,
                            _rho[2].data, _p[2].data, _u[2].data, _B[2].data,
                            _rho[1].data, _p[1].data, _u[1].data, _B[1].data,
                            )
        evt.wait()

        self._ghost_points(_rho[1], _p[1], _u[1], _B[1])

        self.rho_gpu = self.rk1_gpu
        self.u_gpu = self.uk1_gpu
        self.B_gpu = self.Bk1_gpu
        self.p_gpu = self.pk1_gpu

    def _ghost_points(self, _rho, _p, _u, _B):
        shapes = [
            (2*self.config.ghosts, self.config.shape[1], self.config.shape[2]),
            (self.config.shape[0], 2*self.config.ghosts, self.config.shape[2]),
            (self.config.shape[0], self.config.shape[1], 2*self.config.ghosts), ]
        
        for i, _shape in enumerate(shapes):
            evt = self.knl_ghosts(self.queue, _shape, None, np.int32(i), 
                                  _rho.data, _p.data, _u.data, _B.data)
            evt.wait()

    def initials(self):
        print(self.config.true_shape)
        evt = self.knl_initial(self.queue, self.config.true_shape, None, 
                         self.rho_gpu.data, self.p_gpu.data, self.u_gpu.data, self.B_gpu.data)
        evt.wait()
        self._ghost_points(self.rho_gpu, self.p_gpu, self.u_gpu, self.B_gpu)
        self.save_file(0)
