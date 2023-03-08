import os
import numpy as np
import pyopencl as cl
import pyopencl.array
from pathlib import Path
import scipy.stats as stats

import matplotlib.pyplot as plt

from config import Config
from data_service import DataService

class MHDSolver():
    queue = None
    context = None

    config = None
    data_service = None

    knl_kin_e = None
    knl_mag_e = None

    knl_solve = None
    knl_ghosts = None
    knl_initial = None

    p_gpu = None
    u_gpu = None
    B_gpu = None
    rho_gpu = None

    kin_energy = []
    mag_energy = []

    curr_step = 0

    def __init__(self, context) -> None:
        self.context = context
        self.queue = cl.CommandQueue(self.context)

        self._env_vars()
        self._read_config()
        self._build_sources()
        self._init_device_data()
        self.data_service = DataService(rw_energy=self.config.rewrite_energy)


    def _env_vars(self):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        os.environ['PYOPENCL_NO_CACHE'] = '1'
        os.environ['PYOPENCL_CTX'] = '1'


    def _read_config(self):
        self.config = Config()


    def _init_device_data(self):
        self.p_gpu = cl.array.to_device(self.queue, ary=np.zeros(self.config.shape).astype(np.float64))
        self.u_gpu = cl.array.to_device(self.queue, ary=np.zeros(self.config.v_shape).astype(np.float64))
        self.B_gpu = cl.array.to_device(self.queue, ary=np.zeros(self.config.v_shape).astype(np.float64))
        self.rho_gpu = cl.array.to_device(self.queue, ary=np.zeros(self.config.shape).astype(np.float64))

    
    def initials(self):
        print(self.config.true_shape)
        evt = self.knl_initial(self.queue, self.config.true_shape, None, 
                         self.rho_gpu.data, self.p_gpu.data, self.u_gpu.data, self.B_gpu.data)
        evt.wait()
        self._ghost_points(self.rho_gpu, self.p_gpu, self.u_gpu, self.B_gpu)
        self.save_file(0)

    def _build_sources(self):
        c_sources_dir = (Path(__file__).parent / './c_sources')
        with open( (c_sources_dir / './main.cl'), 'r') as file:
            data = file.read()

        prg = cl.Program(self.context, data).build(options=['-I', 
                                                            str(c_sources_dir)])

        self.knl_kin_e = prg.kin_energy
        self.knl_mag_e = prg.mag_energy
        self.knl_solve = prg.solver_3D_RK
        self.knl_ghosts = prg.ghost_nodes_periodic
        self.knl_initial = prg.Orszag_Tang_3D_inital

    
    def read_file(self, i):
        self.data_service.read_data(i, (self.rho_gpu, self.p_gpu, self.u_gpu, self.B_gpu))


    def save_file(self, i):
        self.data_service.save_data(i, (self.rho_gpu, self.p_gpu, self.u_gpu, self.B_gpu))


    def solve(self):
        t = 0
        self.curr_step = self.config.start_step

        if self.config.start_step == 0:
            self.initials()

            self.kin_energy.append(self.compute_kin_energy(self.curr_step))
            self.mag_energy.append(self.compute_mag_energy(self.curr_step))
        else:
            self.read_file(self.config.start_step)

        print('Start solving...')
        while self.curr_step < (self.config.steps + self.config.start_step):
            dT = 0.5 * ( min(self.config.domain_size) / max(self.config.true_shape))**2

            t += dT
            self.curr_step += 1
            self._step(dT)

            if self.curr_step % self.config.rw_del == 0:
                print(f"Step: {self.curr_step}, t: {t}")
                print(f'Writing step_{self.curr_step} file..')
                self.save_file(self.curr_step)

                self.kin_energy.append(self.compute_kin_energy(self.curr_step))
                self.mag_energy.append(self.compute_mag_energy(self.curr_step))

                print(f'Complete!')

        self._save_energy()


    def _step(self, dT):
        pk1_gpu = cl.array.empty(self.queue, self.p_gpu.shape, dtype=np.float64)
        rk1_gpu = cl.array.empty(self.queue, self.rho_gpu.shape, dtype=np.float64)
        uk1_gpu = cl.array.empty(self.queue, self.u_gpu.shape, dtype=np.float64)
        Bk1_gpu = cl.array.empty(self.queue, self.B_gpu.shape, dtype=np.float64)

        pk2_gpu = cl.array.empty(self.queue, self.p_gpu.shape, dtype=np.float64)
        rk2_gpu = cl.array.empty(self.queue, self.rho_gpu.shape, dtype=np.float64)
        uk2_gpu = cl.array.empty(self.queue, self.u_gpu.shape, dtype=np.float64)
        Bk2_gpu = cl.array.empty(self.queue, self.B_gpu.shape, dtype=np.float64)

        _p = [self.p_gpu, pk1_gpu, pk2_gpu]
        _u = [self.u_gpu, uk1_gpu, uk2_gpu]
        _B = [self.B_gpu, Bk1_gpu, Bk2_gpu]
        _rho = [self.rho_gpu, rk1_gpu, rk2_gpu]

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

        self.rho_gpu = rk1_gpu
        self.u_gpu = uk1_gpu
        self.B_gpu = Bk1_gpu
        self.p_gpu = pk1_gpu


    def _ghost_points(self, rho=rho_gpu, p=p_gpu, u=u_gpu, B=B_gpu):
        shapes = [
            (2*self.config.ghosts, self.config.shape[1], self.config.shape[2]),
            (self.config.shape[0], 2*self.config.ghosts, self.config.shape[2]),
            (self.config.shape[0], self.config.shape[1], 2*self.config.ghosts), ]
        
        for i, _shape in enumerate(shapes):
            evt = self.knl_ghosts(self.queue, _shape, None, np.int32(i), 
                                  rho.data, p.data, u.data, B.data)
            evt.wait()


    def get_energy_spectrum(self, energy_gpu):
        fourier_image = np.fft.fftn(energy_gpu.get())
        fourier_amplitudes = np.abs(fourier_image)**2

        kfreq_0 = np.fft.fftfreq(self.config.true_shape[0])\
            * self.config.domain_size[0]
        kfreq_1 = np.fft.fftfreq(self.config.true_shape[1])\
            * self.config.domain_size[1]
        kfreq_2 = np.fft.fftfreq(self.config.true_shape[2])\
            * self.config.domain_size[2]

        kfreq3D = np.meshgrid(kfreq_0, kfreq_1, kfreq_2)
        print(np.max(fourier_amplitudes), np.min(fourier_amplitudes))
        knrm = np.sqrt(kfreq3D[0]**2 + kfreq3D[1]**2 + kfreq3D[2]**2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, np.mean(self.config.true_shape[0])//2+1, 1.)

        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic = "sum",
                                            bins = kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

        kol = -(5.0/3.0)
        # kol = -10
        def kolmogorov(x):
            return np.power(x, kol)
        
        def kr_yor(x):
            return np.power(x, -(3.0/2.0))
        
        k_sprec = np.vectorize(kolmogorov)
        kr_yor_spec = np.vectorize(kr_yor)

        Y = k_sprec(kvals)
        Y_kr = kr_yor_spec(kvals)

        plt.loglog(kvals, Abins)
        # plt.loglog(kvals, Y)
        # plt.loglog(kvals, Y_kr)

        plt.show()
        plt.cla()
        return kvals, Abins

        # bins = 1.
        # tbins = self.config.true_shape[0]

        # density = energy_gpu.get()

        # x, y, z = np.mgrid[0:self.config.true_shape[0], 
        #     0:self.config.true_shape[0], 
        #     0:self.config.true_shape[0]]

        # dist = np.sqrt(x**2+y**2+z**2)

        # FT = np.fft.fftn(density)
        # power = FT.real*FT.real + FT.imag*FT.imag

        # P = power.reshape(np.size(power))
        # dist = dist.reshape(np.size(dist))

        # intervals = np.array([nn*bins for nn in range(0,int(tbins)+1)])

        # p = np.histogram(dist, bins=intervals, weights=P)[0]
        # pd = np.histogram(dist, bins=intervals)[0]
        # pd.astype('float')
        # p = p/pd

        # plt.figure()
        # plt.plot(2.*np.pi/intervals[1:], p)
        # plt.show()


    def _save_energy(self):
        self.kin_energy = np.array(self.kin_energy).astype(np.float64)
        self.mag_energy = np.array(self.mag_energy).astype(np.float64)

        self.data_service.save_energy((self.kin_energy, self.mag_energy))


    def compute_kin_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)

        kin_energy_gpu = cl.array.empty(self.queue, 
                                        self.config.true_shape, dtype=np.float64)

        evt = self.knl_kin_e(self.queue, self.config.true_shape, None, 
                             self.rho_gpu.data, self.u_gpu.data, kin_energy_gpu.data)
        evt.wait()

        self.get_energy_spectrum(kin_energy_gpu)

        return cl.array.sum(kin_energy_gpu).get()


    def compute_mag_energy(self, i):
        if self.curr_step != i:
            self.read_file(i)

        mag_energy_gpu = cl.array.empty(self.queue, 
                                        self.config.true_shape, dtype=np.float64)

        evt = self.knl_mag_e(self.queue, 
                             self.config.true_shape, None, self.B_gpu.data, mag_energy_gpu.data)
        evt.wait()

        self.get_energy_spectrum(mag_energy_gpu)

        return cl.array.sum(mag_energy_gpu).get()
