import os
import re
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

    t = 0
    kin_energy = []
    mag_energy = []
    time_energy = []

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


    def _replace_define(self, src, define_name, value):
        reg_str = f'#define {define_name} [a-zA-Z0-9_.-]*'
        replace_str = f'#define {define_name} {value}'
        
        return re.sub(reg_str, replace_str, src)


    def _build_sources(self):
        print('Building cl sources...')
        c_sources_dir = (Path(__file__).parent / './c_sources')

        with open((c_sources_dir / 'common/mhd_consts_base.cl'), 'r') as file :
            src = file.read()

        for name, value in self.config.defines:
            src = self._replace_define(src, name, value)

        with open((c_sources_dir / 'common/mhd_consts.cl'), 'w') as file:
            file.write(src)

        with open((c_sources_dir / 'main.cl'), 'r') as file:
            data = file.read()

        prg = cl.Program(self.context, data).build(options=['-I', 
                                                            str(c_sources_dir)])

        self.knl_kin_e = prg.kin_energy
        self.knl_mag_e = prg.mag_energy
        self.knl_solve = prg.solver_3D_RK
        self.knl_ghosts = prg.ghost_nodes_periodic
        self.knl_initial = prg.Orszag_Tang_3D_inital

        print('Building is done!')

    
    def read_file(self, i):
        self.t, rho_, p_, u_, B_ = self.data_service.read_data(i)

        cl.enqueue_copy(self.queue, self.rho_gpu.data, rho_[:])
        cl.enqueue_copy(self.queue, self.p_gpu.data, p_[:])
        cl.enqueue_copy(self.queue, self.u_gpu.data, u_[:])
        cl.enqueue_copy(self.queue, self.B_gpu.data, B_[:])


    def save_file(self, i):
        self.data_service.save_data(i, (self.t, self.rho_gpu, self.p_gpu, self.u_gpu, self.B_gpu))


    def compute_energy_only(self):
        print('Start computing...')
        self.curr_step = self.config.start_step
        while self.curr_step <= (self.config.steps + self.config.start_step):
            print(f"Step: {self.curr_step}:")
            self.read_file(self.curr_step)

            self.kin_energy.append(self.compute_kin_energy(self.curr_step))
            self.mag_energy.append(self.compute_mag_energy(self.curr_step))
            self.time_energy.append(self.t)
            self.curr_step += self.config.rw_del
            print(f'Complete!')

        self._save_energy()


    def plot_energy_spectrums_only(self):
        print('Start plotting...')
        self.curr_step = self.config.start_step
        while self.curr_step <= (self.config.steps + self.config.start_step):
            print(f"Step: {self.curr_step}:")
            self.read_file(self.curr_step)

            self.get_kin_energy_spectrum(self.curr_step)
            self.get_mag_energy_spectrum(self.curr_step)
            self.curr_step += self.config.rw_del
            print(f'Complete!')


    def get_energy_spec(self, idx):
        self.curr_step = idx
        self.read_file(self.curr_step)

        kin_k, kin_a = self.get_kin_energy_spectrum(self.curr_step)
        mag_k, mag_A = self.get_mag_energy_spectrum(self.curr_step)
        return (mag_k, mag_A), (kin_k, kin_a)


    def solve(self):
        self.curr_step = self.config.start_step

        self.t = 0
        if self.config.start_step == 0:
            print('Solve initials...')
            self.initials()

            self.kin_energy.append(self.compute_kin_energy(self.curr_step))
            self.mag_energy.append(self.compute_mag_energy(self.curr_step))
        else:
            self.read_file(self.config.start_step)

        print('Start solving...')
        while self.curr_step < (self.config.steps + self.config.start_step):
            dT = 0.5 * ( min(self.config.domain_size) / max(self.config.true_shape))**2

            self.t += dT
            self.curr_step += 1
            self._step(dT)

            if self.curr_step % self.config.rw_del == 0:
                print(f"Step: {self.curr_step}, t: {self.t}")
                print(f'Writing step_{self.curr_step} file..')
                self.save_file(self.curr_step)

                self.kin_energy.append(self.compute_kin_energy(self.curr_step))
                self.mag_energy.append(self.compute_mag_energy(self.curr_step))
                self.time_energy.append(self.t)

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


    def get_kin_energy_spectrum(self, step):
        if step != self.curr_step:
            self.read_file(step)

        print("Start computing kin spectrum:")
        _, kvals, Abins = self.compute_tke_spectrum_3D(
            self.rho_gpu.get(),
            self.u_gpu.get()
        )
        print("Start computing kin spectrum: finished")
        path = self.data_service.get_or_create_dir('graphs/kin_e/')
        plt.loglog(kvals, Abins)
        plt.loglog(kvals, Abins,'ro')
        
        plt.ylabel('Спектр кинетической энергии')
        plt.xlabel('k')

        # plt.show()
        plt.savefig(path / f'./kin_e_spectrum_{step}.jpg')
        plt.cla()
        return kvals, Abins
    

    def get_mag_energy_spectrum(self, step):
        if step != self.curr_step:
            self.read_file(step)

        print("Start computing mag spectrum:")
        _, kvals, Abins = self.compute_tme_spectrum_3D(
            self.B_gpu.get()
        )
        print("Start computing mag spectrum: finished")
        
        path = self.data_service.get_or_create_dir('graphs/mag_e/')
        plt.loglog(kvals, Abins)
        plt.loglog(kvals, Abins,'ro')

        plt.ylabel('Спектр магнитной энергии')
        plt.xlabel('k')

        # plt.show()
        plt.savefig(path / f'./mag_e_spectrum_{step}.jpg')
        plt.cla()
        return kvals, Abins


    def _save_energy(self):
        self.kin_energy = np.array(self.kin_energy).astype(np.float64)
        self.mag_energy = np.array(self.mag_energy).astype(np.float64)
        self.time_energy = np.array(self.time_energy).astype(np.float64)

        self.data_service.save_energy((self.kin_energy, self.mag_energy, self.time_energy))


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


    def compute_tke_spectrum_3D(self, rho, u_gpu, smooth=False):
        import numpy as np
        from numpy.fft import fftn
        from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

        lx = self.config.domain_size[0]
        ly = self.config.domain_size[0]
        lz = self.config.domain_size[0]

        _ghosts = self.config.ghosts
        u = u_gpu[0][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]
        v = u_gpu[1][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]
        w = u_gpu[2][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]

        nx = self.config.true_shape[0]
        ny = self.config.true_shape[1]
        nz = self.config.true_shape[2]

        nt = nx * ny * nz
        n = nx  # int(np.round(np.power(nt,1.0/3.0)))

        # rh = fftn(rho) / nt
        uh = fftn(u) / nt
        vh = fftn(v) / nt
        wh = fftn(w) / nt

        tkeh = 0.5 * ( (uh * conj(uh) + vh * conj(vh) + wh * conj(wh)) ).real

        k0x = 2.0 * pi / lx
        k0y = 2.0 * pi / ly
        k0z = 2.0 * pi / lz

        knorm = (k0x + k0y + k0z) / 3.0
        # print('knorm = ', knorm)

        kxmax = nx / 2
        kymax = ny / 2
        kzmax = nz / 2

        # dk = (knorm - kmax)/n
        # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

        wave_numbers = knorm * arange(0, n)

        tke_spectrum = zeros(len(wave_numbers))

        for kx in range(-nx//2, nx//2-1):
            for ky in range(-ny//2, ny//2-1):
                for kz in range(-nz//2, nz//2-1):
                    rk = sqrt(kx**2 + ky**2 + kz**2)
                    k = int(np.round(rk))
                    tke_spectrum[k] += tkeh[kx, ky, kz]
        # for kx in range(nx):
        #     rkx = kx
        #     if kx > kxmax:
        #         rkx = rkx - nx
        #     for ky in range(ny):
        #         rky = ky
        #         if ky > kymax:
        #             rky = rky - ny
        #         for kz in range(nz):
        #             rkz = kz
        #             if kz > kzmax:
        #                 rkz = rkz - nz
        #             rk = sqrt(rkx * rkx + rky * rky + rkz * rkz)
        #             k = int(np.round(rk))
        #             tke_spectrum[k] = tke_spectrum[k] + tkeh[kx, ky, kz]

        tke_spectrum = tke_spectrum / knorm

        #  tke_spectrum = tke_spectrum[1:]
        #  wave_numbers = wave_numbers[1:]
        # if smooth:
        #     tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        #     tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        #     tke_spectrum = tkespecsmooth

        knyquist = knorm * min(nx, ny, nz) / 2

        return knyquist, wave_numbers[:], tke_spectrum[:]
    

    def compute_tke_spectrum_2D(self, rho, u_gpu, smooth=False):
        import numpy as np
        from numpy.fft import fftn
        from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

        lx = self.config.domain_size[0]
        ly = self.config.domain_size[0]

        _ghosts = self.config.ghosts
        u = u_gpu[0][_ghosts:-_ghosts, _ghosts:-_ghosts, _ghosts:-_ghosts]
        v = u_gpu[1][_ghosts:-_ghosts, _ghosts:-_ghosts, _ghosts:-_ghosts]

        nx = self.config.true_shape[0]
        ny = self.config.true_shape[1]

        nt = nx * ny
        n = nx  # int(np.round(np.power(nt,1.0/3.0)))

        # rh = fftn(rho) / nt
        uh = fftn(u) / nt
        vh = fftn(v) / nt

        tkeh = 0.5 * ( (uh * conj(uh) + vh * conj(vh)) ).real

        k0x = 2.0 * pi / lx
        k0y = 2.0 * pi / ly

        knorm = (k0x + k0y) / 3.0
        # print('knorm = ', knorm)

        kxmax = nx / 2
        kymax = ny / 2

        # dk = (knorm - kmax)/n
        # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

        wave_numbers = knorm * arange(0, n)

        tke_spectrum = zeros(len(wave_numbers))

        for kx in range(-nx//2, nx//2-1):
            for ky in range(-ny//2, ny//2-1):
                rk = sqrt(kx**2 + ky**2)
                k = int(np.round(rk))
                tke_spectrum[k] += tkeh[kx, ky, 0]
     

        tke_spectrum = tke_spectrum / knorm

        knyquist = knorm * min(nx, ny) / 2

        return knyquist, wave_numbers[:], tke_spectrum[:]


    def compute_tme_spectrum_3D(self, B_gpu, smooth=False):
        import numpy as np
        from numpy.fft import fftn
        from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

        lx = self.config.domain_size[0]
        ly = self.config.domain_size[0]
        lz = self.config.domain_size[0]

        _ghosts = self.config.ghosts
        u = B_gpu[0][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]
        v = B_gpu[1][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]
        w = B_gpu[2][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]

        nx = self.config.true_shape[0]
        ny = self.config.true_shape[1]
        nz = self.config.true_shape[2]

        nt = nx * ny * nz
        n = nx  # int(np.round(np.power(nt,1.0/3.0)))

        uh = fftn(u) / nt
        vh = fftn(v) / nt
        wh = fftn(w) / nt

        tkeh = 0.5 * (uh * conj(uh) + vh * conj(vh) + wh * conj(wh)).real

        k0x = 2.0 * pi / lx
        k0y = 2.0 * pi / ly
        k0z = 2.0 * pi / lz

        knorm = (k0x + k0y + k0z) / 3.0
        # print('knorm = ', knorm)

        kxmax = nx / 2
        kymax = ny / 2
        kzmax = nz / 2

        # dk = (knorm - kmax)/n
        # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

        wave_numbers = knorm * arange(0, n)

        tke_spectrum = zeros(len(wave_numbers))

        for kx in range(-nx//2, nx//2-1):
            for ky in range(-ny//2, ny//2-1):
                for kz in range(-nz//2, nz//2-1):
                    rk = sqrt(kx**2 + ky**2 + kz**2)
                    k = int(np.round(rk))
                    tke_spectrum[k] += tkeh[kx, ky, kz]
        # for kx in range(nx):
        #     rkx = kx
        #     if kx > kxmax:
        #         rkx = rkx - nx
        #     for ky in range(ny):
        #         rky = ky
        #         if ky > kymax:
        #             rky = rky - ny
        #         for kz in range(nz):
        #             rkz = kz
        #             if kz > kzmax:
        #                 rkz = rkz - nz
        #             rk = sqrt(rkx * rkx + rky * rky + rkz * rkz)
        #             k = int(np.round(rk))
        #             tke_spectrum[k] = tke_spectrum[k] + tkeh[kx, ky, kz]

        tke_spectrum = tke_spectrum / knorm

        #  tke_spectrum = tke_spectrum[1:]
        #  wave_numbers = wave_numbers[1:]
        # if smooth:
        #     tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        #     tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        #     tke_spectrum = tkespecsmooth

        knyquist = knorm * min(nx, ny, nz) / 2

        return knyquist, wave_numbers[:], tke_spectrum[:]
    

    def compute_tme_spectrum_2D(self, B_gpu, smooth=False):
        import numpy as np
        from numpy.fft import fftn
        from numpy import sqrt, zeros, conj, pi, arange, ones, convolve

        lx = self.config.domain_size[0]
        ly = self.config.domain_size[0]

        _ghosts = self.config.ghosts
        u = B_gpu[0][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]
        v = B_gpu[1][_ghosts:-_ghosts,_ghosts:-_ghosts,_ghosts:-_ghosts]

        nx = self.config.true_shape[0]
        ny = self.config.true_shape[1]

        nt = nx * ny
        n = nx  # int(np.round(np.power(nt,1.0/3.0)))

        uh = fftn(u) / nt
        vh = fftn(v) / nt

        tkeh = 0.5 * (uh * conj(uh) + vh * conj(vh)).real

        k0x = 2.0 * pi / lx
        k0y = 2.0 * pi / ly

        knorm = (k0x + k0y) / 3.0
        # print('knorm = ', knorm)

        kxmax = nx / 2
        kymax = ny / 2

        # dk = (knorm - kmax)/n
        # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

        wave_numbers = knorm * arange(0, n)

        tke_spectrum = zeros(len(wave_numbers))

        for kx in range(-nx//2, nx//2-1):
            for ky in range(-ny//2, ny//2-1):
                rk = sqrt(kx**2 + ky**2)
                k = int(np.round(rk))
                tke_spectrum[k] += tkeh[kx, ky, 0]


        tke_spectrum = tke_spectrum / knorm


        knyquist = knorm * min(nx, ny) / 2

        return knyquist, wave_numbers[:], tke_spectrum[:]


    def plot_energy(self):
        print('Start plotting energy...')
        kin_e, mag_e = self.data_service.get_energy()

        dT = 0.5 * ( min(self.config.domain_size) / max(self.config.true_shape))**2

        t = dT * np.array(list(range(len(kin_e))))

        path = self.data_service.get_or_create_dir('graphs/kin_e/')
        plt.plot(t, kin_e)
        plt.plot(t, kin_e,'ro')

        plt.ylabel('Кинетическая энергия')
        plt.xlabel('t')

        print('Kin energy finished!')
        # plt.show()
        plt.savefig(path / f'./kin_e.jpg')
        plt.cla()

        path = self.data_service.get_or_create_dir('graphs/mag_e/')
        plt.plot(t, mag_e)
        plt.plot(t, mag_e,'ro')

        plt.ylabel('Магнитная энергия')
        plt.xlabel('t')

        print('Mag energy finished!')

        # plt.show()
        plt.savefig(path / f'./mag_e.jpg')
        plt.cla()


    def plot_fields(self):
        print('Start plotting...')
        self.curr_step = self.config.start_step
        while self.curr_step <= (self.config.steps + self.config.start_step):
            print(f"Step: {self.curr_step}:")
            self.read_file(self.curr_step)

            self.plot_rho()
            for i in range(3):
                self.plot_u(ax=i)
                self.plot_B(ax=i)

            self.curr_step += self.config.rw_del
            print(f'Complete!')

    def plot_rho(self, z=0):
        path = self.data_service.get_or_create_dir('graphs/rho/')
        plt.xlim(0, self.config.domain_size[0] * (self.config.shape[0]/self.config.true_shape[0]))
        plt.ylim(0, self.config.domain_size[1] * (self.config.shape[1]/self.config.true_shape[1]))

        x = np.linspace(0, self.config.domain_size[0] \
                    * (self.config.shape[0]/self.config.true_shape[0]), self.config.shape[0])
        y = np.linspace(0, self.config.domain_size[1] \
                    * (self.config.shape[1]/self.config.true_shape[1]), self.config.shape[1])
        # X, Y = np.meshgrid(x, y)

        plt.contourf(x,y, self.rho_gpu.get()[:, :, z], levels = 20, cmap='plasma')
        plt.colorbar(label='rho')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(path / f'./rho_{self.curr_step}.jpg')
        plt.cla()
        plt.clf()

    def _ax_to_сhar(self, ax):
        if ax == 0:
            return 'x'
        elif ax == 1:
            return 'y'
        elif ax == 2:
            return 'z'
        else:
            return '_'

    def plot_u(self, ax=0, z=0):
        path = self.data_service.get_or_create_dir(f'graphs/u/u_{self._ax_to_сhar(ax)}/')
        plt.xlim(0, self.config.domain_size[0] * (self.config.shape[0]/self.config.true_shape[0]))
        plt.ylim(0, self.config.domain_size[1] * (self.config.shape[1]/self.config.true_shape[1]))

        x = np.linspace(0, self.config.domain_size[0] \
                    * (self.config.shape[0]/self.config.true_shape[0]), self.config.shape[0])
        y = np.linspace(0, self.config.domain_size[1] \
                    * (self.config.shape[1]/self.config.true_shape[1]), self.config.shape[1])
        # x, y = np.meshgrid(x, y)

        plt.contourf(x,y, self.u_gpu.get()[ax, :, :, z], levels = 20, cmap='plasma')
        plt.colorbar(label=f'u_{self._ax_to_сhar(ax)}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(path / f'./u{self._ax_to_сhar(ax)}_{self.curr_step}.jpg')
        plt.cla()
        plt.clf()

    def plot_B(self, ax=0, z=0):
        path = self.data_service.get_or_create_dir(f'graphs/B/B_{self._ax_to_сhar(ax)}/')
        plt.xlim(0, self.config.domain_size[0] * (self.config.shape[0]/self.config.true_shape[0]))
        plt.ylim(0, self.config.domain_size[1] * (self.config.shape[1]/self.config.true_shape[1]))

        x = np.linspace(0, self.config.domain_size[0] \
                    * (self.config.shape[0]/self.config.true_shape[0]), self.config.shape[0])
        y = np.linspace(0, self.config.domain_size[1] \
                    * (self.config.shape[1]/self.config.true_shape[1]), self.config.shape[1])
        # x, y = np.meshgrid(x, y)

        plt.contourf(x,y, self.B_gpu.get()[ax, :, :, z], levels = 20, cmap='plasma')
        plt.colorbar(label=f'B_{self._ax_to_сhar(ax)}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(path / f'./B_{self._ax_to_сhar(ax)}_{self.curr_step}.jpg')
        plt.cla()
        plt.clf()
