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

        kfreq_0 = np.fft.fftfreq(self.config.true_shape[0], d=\
            self.config.domain_size[0])
        kfreq_1 = np.fft.fftfreq(self.config.true_shape[1], d=\
            self.config.domain_size[1])
        kfreq_2 = np.fft.fftfreq(self.config.true_shape[2], d=\
            self.config.domain_size[2])

        kfreq3D = np.meshgrid(kfreq_0, kfreq_1, kfreq_2)
        knrm = np.sqrt(kfreq3D[0]**2 + kfreq3D[1]**2 + kfreq3D[2]**2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, np.mean(self.config.true_shape[0] * self.config.domain_size[0])//2+1, 1.)

        kvals = 0.5 * (kbins[1:] + kbins[:-1])

        print(kbins)
        print(fourier_amplitudes)
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic = "sum",
                                            bins = kbins)
        # Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

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

        _, kvals, Abins = compute_tke_spectrum(energy_gpu.get(),  
            self.config.domain_size[0],
             self.config.domain_size[1],
              self.config.domain_size[2])

        plt.loglog(kvals, Abins)
        # plt.loglog(kvals, Y)
        # plt.loglog(kvals, Y_kr)

        plt.show()
        plt.cla()
        return kvals, Abins


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


def compute_tke_spectrum(u,lx,ly,lz,smooth=False):

    from numpy.fft import fftn
    from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

    """
    Given a velocity field u, v, w, this function computes the kinetic energy
    spectrum of that velocity field in spectral space. This procedure consists of the 
    following steps:
    1. Compute the spectral representation of u, v, and w using a fast Fourier transform.
    This returns uf, vf, and wf (the f stands for Fourier)
    2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf, vf, wf)* conjugate(uf, vf, wf)
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy 
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).

    Parameters:
    -----------  
    u: 3D array
        The x-velocity component.
    v: 3D array
        The y-velocity component.
    w: 3D array
        The z-velocity component.    
    lx: float
        The domain size in the x-direction.
    ly: float
        The domain size in the y-direction.
    lz: float
        The domain size in the z-direction.
    smooth: boolean
        A boolean to smooth the computed spectrum for nice visualization.
    """
    nx = len(u[:,0,0])
    ny = len(u[0,:,0])
    nz = len(u[0,0,:])
    
    nt= nx*ny*nz
    n = nx #int(np.round(np.power(nt,1.0/3.0)))
    
    uh = fftn(u)/nt
    
    tkeh = zeros((nx,ny,nz))
    tkeh = (uh*conj(uh)).real
    
    k0x = 2.0*pi/lx
    k0y = 2.0*pi/ly
    k0z = 2.0*pi/lz
    
    knorm = (k0x + k0y + k0z)/3.0
    
    kxmax = nx/2
    kymax = ny/2
    kzmax = nz/2
    
    wave_numbers = knorm*arange(0,n)
    
    tke_spectrum = zeros(len(wave_numbers))
    
    for kx in xrange(nx):
        rkx = kx
        if (kx > kxmax):
            rkx = rkx - (nx)
        for ky in xrange(ny):
            rky = ky
            if (ky>kymax):
                rky=rky - (ny)
            for kz in xrange(nz):        
                rkz = kz
                if (kz>kzmax):
                    rkz = rkz - (nz)
                rk = sqrt(rkx*rkx + rky*rky + rkz*rkz)
                k = int(np.round(rk))
                tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky,kz]

    tke_spectrum = tke_spectrum/knorm
    #  tke_spectrum = tke_spectrum[1:]
    #  wave_numbers = wave_numbers[1:]
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm*min(nx,ny,nz)/2 

    return knyquist, wave_numbers, tke_spectrum