from math import prod

import taichi as ti
import taichi.math as tm

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from config import Config
from logger import Logger
from cl_builder import CLBuilder
from data_service import DataService

from taichi_src.kernels.sgs_constants.sgs_constants_kernels import *
        
def get_filtered_shape(shape, filter_size):
    ret = list(shape)
    for i in range(-3, 0):
        ret[i] //= filter_size
    return tuple(ret)

class MHD_Solver:
    def __init__(self, context, config, data_path=''):
        self.C = 0
        self.Y = 0
        self.D = 0

        self.config = config

        self.context = context
        self.queue = cl.CommandQueue(self.context)
        self.program = CLBuilder.build(self.context, self.config.defines)

        self._init_device_data()
        self._define_kernels()

        self.data_service = DataService(dir_name=data_path,rw_energy=self.config.rewrite_energy)

        ti.init(arch=ti.vulkan, debug=True)
        self.init_taichi_arrs()

    def _define_kernels(self):
        self.knl_kin_e = self.program.kin_energy
        self.knl_mag_e = self.program.mag_energy
        self.knl_solve = self.program.solver_3D_RK
        self.knl_ghosts = self.program.ghost_nodes_periodic
        self.knl_initial = self.program.Orszag_Tang_3D_inital

        self.knl_get_J = self.program.get_J
        self.knl_get_S = self.program.get_S
        self.knl_get_phi = self.program.get_phi
        self.knl_get_alpha = self.program.get_alpha

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

        self.S_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.mat_shape).astype(np.float64))
        self.J_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.mat_shape).astype(np.float64))
        self.phi_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.mat_shape).astype(np.float64))
        self.alpha_gpu = cla.to_device(self.queue, ary=np.zeros(self.config.mat_shape).astype(np.float64))

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

            if self.config.model.lower() != 'dns':
                self.compute_coefs()

            if self.current_step % self.config.rw_del == 0:
                self.save_file(self.current_step)
                
    def _get_idx(self, x, y, z):
        return self.config.shape[0]*self.config.shape[1] * x + self.config.shape[1] * y + z
    
    def _get_v_idx(self, ax, x, y, z):
        return self.config.shape[0]*self.config.shape[1]*self.config.shape[2]*ax +\
        self.config.shape[0]*self.config.shape[1] * x + self.config.shape[1] * y + z
    
    def _get_list_idx(self, x0, x1, y0, y1, z0, z1):
        ret = []
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    ret.append(self._get_idx(x, y, z))

        return np.array(ret, dtype=np.int32)
    
    def _get_list_v_idx(self, ax, x0, x1, y0, y1, z0, z1):
        ret = []
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    ret.append(self._get_v_idx(ax, x, y, z))

        return np.array(ret, dtype=np.int32)

    def init_taichi_arrs(self):
        self.Ma = 1.1
        self.FILTER_SIZE = 2

        self.u_ti = ti.ndarray(dtype=ti.float64, shape=self.u_gpu.shape)
        self.B_ti = ti.ndarray(dtype=ti.float64, shape=self.B_gpu.shape)
        self.rho_ti = ti.ndarray(dtype=ti.float64, shape=self.rho_gpu.shape)

        self.S_ti = ti.ndarray(dtype=ti.f64, shape=self.alpha_gpu.shape)
        self.J_ti = ti.ndarray(dtype=ti.f64, shape=self.alpha_gpu.shape)
        self.phi_ti = ti.ndarray(dtype=ti.f64, shape=self.phi_gpu.shape)
        self.alpha_ti = ti.ndarray(dtype=ti.f64, shape=self.alpha_gpu.shape)


        self.u_ti_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.u_gpu.shape, self.FILTER_SIZE))
        self.B_ti_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.B_gpu.shape, self.FILTER_SIZE))
        self.rho_ti_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.rho_gpu.shape, self.FILTER_SIZE))

        self.S_ti_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.alpha_gpu.shape, self.FILTER_SIZE))
        self.J_ti_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.alpha_gpu.shape, self.FILTER_SIZE))
        self.phi_ti_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.phi_gpu.shape, self.FILTER_SIZE))
        self.alpha_ti_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.alpha_gpu.shape, self.FILTER_SIZE))

    def update_ti_arrs(self):
        self.u_ti.from_numpy(self.u_gpu.get())
        self.B_ti.from_numpy(self.B_gpu.get())
        self.rho_ti.from_numpy(self.rho_gpu.get())

        self.S_ti.from_numpy(self.S_gpu.get())
        self.J_ti.from_numpy(self.J_gpu.get())
        self.alpha_ti.from_numpy(self.alpha_gpu.get())
        self.phi_ti.from_numpy(self.phi_gpu.get())

        filter_sc(self.rho_ti, self.rho_ti_filter, self.FILTER_SIZE)
        filter_vec(self.u_ti, self.u_ti_filter, self.FILTER_SIZE)
        filter_vec(self.B_ti, self.B_ti_filter, self.FILTER_SIZE)

        filter_mat(self.S_ti, self.S_ti_filter, self.FILTER_SIZE)
        filter_mat(self.J_ti, self.J_ti_filter, self.FILTER_SIZE)
        filter_mat(self.phi_ti, self.phi_ti_filter, self.FILTER_SIZE)
        filter_mat(self.alpha_ti, self.alpha_ti_filter, self.FILTER_SIZE)

    def get_Lu(self):
        # Lu = ti.ndarray(dtype=ti.float64, shape=(3, 3, ) + self.config.shape)
        Lu_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))


        rho_u_ = ti.ndarray(dtype=ti.f64, shape=self.config.v_shape)
        rho_u_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.v_shape, self.FILTER_SIZE))
        rho_u(self.rho_ti, self.u_ti, rho_u_)

        filter_vec(rho_u_, rho_u_filter, self.FILTER_SIZE)


        Lu_a_ = ti.ndarray(dtype=ti.f64, shape=self.config.mat_shape)
        Lu_A(self.rho_ti, rho_u_, Lu_a_)
        Lu_a_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        filter_mat(Lu_a_, Lu_a_filter, self.FILTER_SIZE)

        Lu_a_ = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        Lu_A(self.rho_ti_filter, rho_u_filter, Lu_a_)

        BiBj_ = ti.ndarray(dtype=ti.f64, shape=self.config.mat_shape)
        BiBj_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))
        BiBj(self.B_ti, BiBj_)
        filter_mat(BiBj_, BiBj_filter, self.FILTER_SIZE)

        BiBj_ = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        BiBj(self.B_ti_filter, BiBj_)        

        Lu_compute(self.Ma, Lu_filter, BiBj_, BiBj_filter, Lu_a_, Lu_a_filter)
        return Lu_filter

    def get_Lb(self):
        # Lb = ti.ndarray(dtype=ti.float64, shape=self.config.mat_shape)
        Lb_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))
        Lb_filter2 = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))


        rho_u_ = ti.ndarray(dtype=ti.f64, shape=self.config.v_shape)
        rho_u_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.v_shape, self.FILTER_SIZE))
        rho_u(self.rho_ti, self.u_ti, rho_u_)

        filter_vec(rho_u_, rho_u_filter, self.FILTER_SIZE)

        Lb_a = ti.ndarray(dtype=ti.f64, shape=self.config.mat_shape)
        Lb_A(self.rho_ti, rho_u_, self.B_ti, Lb_a)

        Lb_a_filtered = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))
        filter_mat(Lb_a, Lb_a_filtered, self.FILTER_SIZE)

        Lb_a = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))
        Lb_A(self.rho_ti, rho_u_filter, self.B_ti_filter, Lb_a)

        Lb = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        Lb_compute(Lb, Lb_a, Lb_a_filtered)
        return Lb

    def get_phi(self):
        evt = self.knl_get_phi(self.queue, self.config.true_shape, None,
                                    self.u_gpu.data, self.B_gpu.data, self.phi_gpu.data)
        evt.wait()


    def get_alpha(self):
        evt = self.knl_get_alpha(self.queue, self.config.true_shape, None,
                                    self.rho_gpu.data, self.u_gpu.data,
                                        self.B_gpu.data, self.alpha_gpu.data)
        evt.wait()

    def get_S(self):
        evt = self.knl_get_S(self.queue, self.config.true_shape, None,
                                    self.u_gpu.data, self.S_gpu.data)
        evt.wait()

    def get_J(self):
        evt = self.knl_get_J(self.queue, self.config.true_shape, None, 
                                self.B_gpu.data, self.J_gpu.data)
        evt.wait()
        

    def get_M(self):
        M_ti = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        M_A_ti = ti.ndarray(dtype=ti.f64, shape=self.config.mat_shape)
        M_A_ti_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        M_A_compute(M_A_ti, self.alpha_ti, self.S_ti)
        filter_mat(M_A_ti, M_A_ti_filter, self.FILTER_SIZE)

        M_A_ti = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))
        M_A_compute(M_A_ti, self.alpha_ti_filter, self.S_ti_filter)

        M_compute(M_ti, M_A_ti, M_A_ti_filter)
        return M_ti

    def get_m(self):
        m_ti = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        m_A_ti = ti.ndarray(dtype=ti.f64, shape=self.config.mat_shape)
        m_A_ti_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))

        m_A_compute(m_A_ti, self.phi_ti, self.J_ti)
        filter_mat(m_A_ti, m_A_ti_filter, self.FILTER_SIZE)

        m_A_ti = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.mat_shape, self.FILTER_SIZE))
        m_A_compute(m_A_ti, self.phi_ti_filter, self.J_ti_filter)

        M_compute(m_ti, m_A_ti, m_A_ti_filter)
        return m_ti

    def compute_coefs(self):
        self.get_S()
        self.get_J()
        self.get_phi()
        self.get_alpha()

        self.update_ti_arrs()

        Lu = self.get_Lu()
        Lb = self.get_Lb()
        
        M = self.get_M()
        m = self.get_m()

        MM = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        MM.fill(0)
        mat_dot(M, M, MM)

        mm = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        mm.fill(0)
        mat_dot(m, m, mm)

        LuM = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        LuM.fill(0)
        mat_dot(Lu, M, LuM)

        Lbm = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        Lbm.fill(0)
        mat_dot(Lb, m, Lbm)

        LuLu = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        LuLu.fill(0)
        mat_trace_dot(Lu, Lu, LuLu)

        @ti.kernel
        def aS_compute(alpha: ti.types.ndarray(), 
                        abs_S: ti.types.ndarray(), 
                        result: ti.types.ndarray()):
            for idx in ti.grouped(alpha):
                i, j, x, y, z = idx
                if i == j:
                    result[x, y, z] = result[x, y, z] + alpha[idx] * abs_S[x, y, z]

        abs_S = ti.ndarray(dtype=ti.f64, shape=self.config.shape)
        abs_S_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        abs_S_compute(self.S_ti, abs_S)
        filter_sc(abs_S, abs_S_filter, self.FILTER_SIZE)

        aS = ti.ndarray(dtype=ti.f64, shape=self.config.shape)
        aS_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        aS_compute(self.alpha_ti, abs_S, aS)
        filter_sc(aS, aS_filter, self.FILTER_SIZE)

        aS = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        aS_compute(self.alpha_ti_filter, abs_S_filter, aS)

        @ti.kernel
        def Y_denominator(A: ti.types.ndarray(), A_filter: ti.types.ndarray(), result: ti.types.ndarray()):
            for idx in ti.grouped(result):
                result[idx] = A[idx] - A_filter[idx]

        Y_d = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.shape, self.FILTER_SIZE))
        Y_denominator(aS, aS_filter, Y_d)


        # MM_mean = ti.field(ti.f64, shape=())
        # mm_mean = ti.field(ti.f64, shape=())
        # Y_d_mean = ti.field(ti.f64, shape=())

        # LuM_mean = ti.field(ti.f64, shape=())
        # LuLu_mean = ti.field(ti.f64, shape=())
        # Lbm_mean = ti.field(ti.f64, shape=())

        # spatial_mean(LuM, LuM_mean, self.config.dV)
        # spatial_mean(MM, MM_mean, self.config.dV)
        # spatial_mean(LuLu, LuLu_mean, self.config.dV)
        # spatial_mean(Y_d, Y_d_mean, self.config.dV)
        # spatial_mean(Lbm, Lbm_mean, self.config.dV)
        # spatial_mean(mm, mm_mean, self.config.dV)

        # LuM_mean = LuM_mean.to_numpy()
        # MM_mean = MM_mean.to_numpy()
        # LuLu_mean = LuLu_mean.to_numpy()
        # Y_d_mean = Y_d_mean.to_numpy()
        # mm_mean = mm_mean.to_numpy()
        # Lbm_mean = Lbm_mean.to_numpy()

        LuM_mean = LuM.to_numpy()
        MM_mean = MM.to_numpy()
        LuLu_mean = LuLu.to_numpy()
        Y_d_mean = Y_d.to_numpy()
        mm_mean = mm.to_numpy()
        Lbm_mean = Lbm.to_numpy()

        

        LuM_mean = LuM_mean.sum()*self.config.dV * self.FILTER_SIZE**3
        MM_mean = MM_mean.sum()*self.config.dV * self.FILTER_SIZE**3
        LuLu_mean = LuLu_mean.sum()*self.config.dV * self.FILTER_SIZE**3
        Y_d_mean = Y_d_mean.sum()*self.config.dV * self.FILTER_SIZE**3
        mm_mean = mm_mean.sum()*self.config.dV * self.FILTER_SIZE**3
        Lbm_mean = Lbm_mean.sum()*self.config.dV * self.FILTER_SIZE**3

        # Logger.log(f"LuM_mean: {LuM_mean}, LuLu_mean: {LuLu_mean}, Lbm_mean: {Lbm_mean}")
        # Logger.log(f"MM_mean: {MM_mean}, Y_d_mean: {Y_d_mean}, mm_mean: {mm_mean}")

        self.C =  LuM_mean / MM_mean
        self.Y =  LuLu_mean / Y_d_mean
        self.D =  Lbm_mean / mm_mean

    def _step(self, dT):
        _p = [self.p_gpu, self.pk1_gpu, self.pk2_gpu]
        _u = [self.u_gpu, self.uk1_gpu, self.uk2_gpu]
        _B = [self.B_gpu, self.Bk1_gpu, self.Bk2_gpu]
        _rho = [self.rho_gpu, self.rk1_gpu, self.rk2_gpu]

        coefs = ((0.0, 1.0), (0.75, 0.25), ((1.0/3.0), (2.0/3.0)),)

        evt = self.knl_solve(self.queue, self.config.true_shape, None,
                            np.float64(dT), np.float64(coefs[0][0]), np.float64(coefs[0][1]),
                            np.float64(self.Y), np.float64(self.C), np.float64(self.D),
                            _rho[2].data, _p[2].data, _u[2].data, _B[2].data,
                            _rho[0].data, _p[0].data, _u[0].data, _B[0].data,
                            _rho[1].data, _p[1].data, _u[1].data, _B[1].data,
                            )
        evt.wait()

        self._ghost_points(_rho[1], _p[1], _u[1], _B[1])

        evt = self.knl_solve(self.queue, self.config.true_shape, None,
                            np.float64(dT), np.float64(coefs[1][0]), np.float64(coefs[1][1]),
                            np.float64(self.Y), np.float64(self.C), np.float64(self.D),
                            _rho[0].data, _p[0].data, _u[0].data, _B[0].data,
                            _rho[1].data, _p[1].data, _u[1].data, _B[1].data,
                            _rho[2].data, _p[2].data, _u[2].data, _B[2].data,
                            )
        evt.wait()

        self._ghost_points(_rho[2], _p[2], _u[2], _B[2])

        evt = self.knl_solve(self.queue, self.config.true_shape, None,
                            np.float64(dT), np.float64(coefs[2][0]), np.float64(coefs[2][1]),
                            np.float64(self.Y), np.float64(self.C), np.float64(self.D),
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

        if self.config.model.lower() != 'dns':
            self.compute_coefs()

        self.save_file(0)
