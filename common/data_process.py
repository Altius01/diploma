import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from logger import Logger
from data_service import DataService
from opencl.cl_builder import CLBuilder

class MHD_DataProcessor:
    def __init__(self, context, config, data_path=''):
        self.config = config

        self.context = context
        self.queue = cl.CommandQueue(self.context)
        self.program = CLBuilder.build(self.context, self.config.defines)
        self.data_service = DataService(dir_name=data_path,rw_energy=self.config.rewrite_energy)
        self._define_kernels()
        self._init_device_data()

        self.kin_energy = []
        self.mag_energy = []
        self.time_energy = []
        self.current_time = 0

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

    def read_file(self, i):
        self.current_time, rho_, p_, u_, B_ = self.data_service.read_data(i)

        cl.enqueue_copy(self.queue, self.rho_gpu.data, rho_[:])
        cl.enqueue_copy(self.queue, self.p_gpu.data, p_[:])
        cl.enqueue_copy(self.queue, self.u_gpu.data, u_[:])
        cl.enqueue_copy(self.queue, self.B_gpu.data, B_[:])

    def compute_kin_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)

        kin_energy_gpu = cl.array.empty(self.queue, 
                                        self.config.true_shape, dtype=np.float64)

        evt = self.knl_kin_e(self.queue, self.config.true_shape, None, 
                             self.rho_gpu.data, self.u_gpu.data, kin_energy_gpu.data)
        evt.wait()

        return cl.array.sum(kin_energy_gpu).get()


    def compute_mag_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)

        mag_energy_gpu = cl.array.empty(self.queue, 
                                        self.config.true_shape, dtype=np.float64)

        evt = self.knl_mag_e(self.queue, 
                             self.config.true_shape, None, self.B_gpu.data, mag_energy_gpu.data)
        evt.wait()

        return cl.array.sum(mag_energy_gpu).get()
    
    def compute_energy_only(self, save_energy=True):
        Logger.log('Start computing energies:')
        self.curr_step = self.config.start_step
        while self.current_time <= self.config.end_time:
            Logger.log(f"Step: {self.curr_step}: start.")
            self.read_file(self.curr_step)

            self.kin_energy.append(self.compute_kin_energy(self.curr_step))
            self.mag_energy.append(self.compute_mag_energy(self.curr_step))
            self.time_energy.append(self.current_time)
            Logger.log(f'Step: {self.curr_step}: done!')
            
            self.curr_step += self.config.rw_del
        if save_energy:
            self._save_energy()
    
    def _save_energy(self):
        self.kin_energy = np.array(self.kin_energy).astype(np.float64)
        self.mag_energy = np.array(self.mag_energy).astype(np.float64)
        self.time_energy = np.array(self.time_energy).astype(np.float64)

        self.data_service.save_energy((self.kin_energy, self.mag_energy, self.time_energy))

    def get_energy(self):
        return self.data_service.get_energy()

