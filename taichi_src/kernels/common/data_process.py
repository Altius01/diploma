import numpy as np
import taichi as ti

from logger import Logger
from data_service import DataService

from taichi_src.kernels.solver.computers import *

@ti.data_oriented
class TiDataProcessor:
    def __init__(self, context, config, data_path=''):
        self.config = config

        self.data_service = DataService(dir_name=data_path,rw_energy=self.config.rewrite_energy)

        self._init_device_data()

        self.kin_energy = []
        self.mag_energy = []
        self.time_energy = []
        self.current_time = 0

    @ti.func
    def _check_ghost(self, shape, idx):
        return (idx < self.config.ghosts) or (idx >= shape - self.config.ghosts)

    @ti.func
    def check_ghost_idx(self, idx):
        result = False

        for i in ti.static(range(idx.n)):
            result = result or self._check_ghost(self.config.shape[i], idx[i])

        return result

    def _init_device_data(self):
        self.u = ti.Vector.field(n=3, dtype=ti.f64, shape=self.config.shape)
        self.B = ti.Vector.field(n=3, dtype=ti.f64, shape=self.config.shape)
        self.p = ti.field(dtype=ti.f64, shape=self.config.shape)
        self.rho = ti.field(dtype=ti.f64, shape=self.config.shape)

    def read_file(self, i):
        self.current_step = i
        self.current_time, rho_, p_, u_, B_ = self.data_service.read_data(i)

        self.rho.from_numpy(rho_)
        self.p.from_numpy(p_)
        self.u.from_numpy(u_)
        self.B.from_numpy(B_)


    @ti.kernel
    def _get_kin_energy(self) -> double:
        result = double(0.)

        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                result += 0.5 * self.rho[idx] * self.u[idx].norm_sqr()
        return result
    
    @ti.kernel
    def _get_mag_energy(self) -> double:
        result = double(0.)

        for idx in ti.grouped(self.B):
            if not self.check_ghost_idx(idx):
                result += 0.5 * self.B[idx].norm_sqr()
        return result

    def compute_kin_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)
        return self._get_kin_energy() * self.config.dV
        

    def compute_mag_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)
        return self._get_mag_energy() * self.config.dV
    
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

