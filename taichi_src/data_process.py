import numpy as np
import taichi as ti

from logger import Logger
from data_service import DataService

from taichi_src.common.types import *
from taichi_src.common.boundaries import *

@ti.data_oriented
class TiDataProcessor:
    def __init__(self, config, data_path=''):
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
        self.u = ti.Vector.field(n=3, dtype=double, shape=self.config.shape)
        self.B = ti.Vector.field(n=3, dtype=double, shape=self.config.shape)
        self.p = ti.field(dtype=double, shape=self.config.shape)
        self.rho = ti.field(dtype=double, shape=self.config.shape)

        self.kin_e_field = ti.field(ti.f32, shape=())
        self.mag_e_field = ti.field(ti.f32, shape=())

    def read_file(self, i):
        self.current_step = i
        self.current_time, rho_, p_, u_, B_ = self.data_service.read_data(i)

        self.rho.from_numpy(rho_)
        self.p.from_numpy(p_)
        self.u.from_numpy(u_)
        self.B.from_numpy(B_)


    @ti.kernel
    def _get_kin_energy(self, s: ti.template()):
        for idx in ti.grouped(self.rho):
            if not self.check_ghost_idx(idx):
                s[None] += ti.cast(0.5 * self.rho[idx] * self.u[idx].norm_sqr(), ti.types.f32)
    
    @ti.kernel
    def _get_mag_energy(self, s: ti.template()):
        for idx in ti.grouped(self.B):
            if not self.check_ghost_idx(idx):
                s[None] += ti.cast(0.5 * self.B[idx].norm_sqr(), ti.types.f32)

    def compute_kin_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)

        self.kin_e_field[None] = 0
        self._get_kin_energy(self.kin_e_field)
        return self.kin_e_field[None] * self.config.dV
        

    def compute_mag_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)

        self.mag_e_field[None] = 0
        self._get_mag_energy(self.mag_e_field)
        return self.mag_e_field[None] * self.config.dV
    
    def compute_energy_only(self, save_energy=True):
        Logger.log('Start computing energies:')
        self.curr_step = self.config.start_step
        while self.current_time <= self.config.end_time:
            Logger.log(f"Step: {self.curr_step}: start.")
            self.read_file(self.curr_step)
            Logger.log(f"   Step: {self.curr_step}: compute_kin_energy start.")
            self.kin_energy.append(self.compute_kin_energy(self.curr_step))
            Logger.log(f"   Step: {self.curr_step}: compute_kin_energy done!")
            Logger.log(f"   Step: {self.curr_step}: compute_mag_energy start.")
            self.mag_energy.append(self.compute_mag_energy(self.curr_step))
            Logger.log(f"   Step: {self.curr_step}: compute_mag_energy done!")
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

