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

@ti.func
def get_sc_idx(vec_idx):
    return [vec_idx[i] for i in range(len(vec_idx)-3, len(vec_idx))]

@ti.kernel
def filter_sc(src: ti.types.ndarray(), out: ti.types.ndarray(), filter_size: ti.i32):
    for i in ti.grouped(out):
        for j in ti.grouped(ti.ndrange(filter_size, filter_size, filter_size)):
            l = i * filter_size + j[:]
            ti.atomic_add(out[i], src[l])
        out[i] /= filter_size**3

@ti.kernel
def filter_vec(src: ti.types.ndarray(), out: ti.types.ndarray(), filter_size: ti.i32):
    for i in ti.grouped(out):
        for j in ti.grouped(ti.ndrange(filter_size, filter_size, filter_size)):
            l = i
            l[1:] = l[1:] * filter_size + j[:]

            ti.atomic_add(out[i], src[l])
        out[i] /= filter_size**3

@ti.kernel
def filter_mat(src: ti.types.ndarray(), out: ti.types.ndarray(), filter_size: ti.i32):
    for i in ti.grouped(out):
        for j in ti.grouped(ti.ndrange(filter_size, filter_size, filter_size)):
            l = i
            l[2:] += l[2:] * filter_size + j[:]
            ti.atomic_add(out[i], src[l])
        out[i] /= filter_size**3
        
def get_filtered_shape(shape, filter_size):
    ret = list(shape)
    for i in range(-3, 0):
        ret[i] //= filter_size
    return tuple(ret)

class MHD_Solver:
    def __init__(self, context, config, data_path=''):
        ti.init(arch=ti.gpu, debug=True)
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
    
    def get_Lu(self, filter_size=8):
        u = ti.ndarray(dtype=ti.float64, shape=self.u_gpu.shape)
        B = ti.ndarray(dtype=ti.float64, shape=self.B_gpu.shape)
        p = ti.ndarray(dtype=ti.float64, shape=self.p_gpu.shape)
        rho = ti.ndarray(dtype=ti.float64, shape=self.rho_gpu.shape)

        u.from_numpy(self.u_gpu.get())
        B.from_numpy(self.B_gpu.get())
        p.from_numpy(self.p_gpu.get())
        rho.from_numpy(self.rho_gpu.get())

        u_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.u_gpu.shape, filter_size))
        B_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.B_gpu.shape, filter_size))
        p_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.p_gpu.shape, filter_size))
        rho_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(self.rho_gpu.shape, filter_size))

        FILTER_SIZE = 8

        filter_sc(p, p_filter, FILTER_SIZE)
        filter_sc(rho, rho_filter, FILTER_SIZE)
        filter_vec(u, u_filter, FILTER_SIZE)
        filter_vec(B, B_filter, FILTER_SIZE)

        @ti.kernel
        def rho_u(rho: ti.types.ndarray(), u: ti.types.ndarray(), result: ti.types.ndarray()):
            for i in ti.grouped(result):
                result[i] = rho[get_sc_idx(i)]*u[i]

        @ti.kernel
        def BiBj(B: ti.types.ndarray(), result: ti.types.ndarray()):
            for idx in ti.grouped(result):
                i, j, x, y, z = idx
                result[idx] = B[i, x, y, z]*B[j, x, y, z]

        @ti.kernel
        def Lu_A(rho: ti.types.ndarray(), rho_u: ti.types.ndarray(), result: ti.types.ndarray()):
            for idx in ti.grouped(result):
                i, j, x, y, z = idx
                result[idx] = (rho_u[i, x, y, z] * rho_u[j, x, y, z]) / rho[get_sc_idx(idx)]

        @ti.kernel
        def Lb_A(rho_u: ti.types.ndarray(), B: ti.types.ndarray(), result: ti.types.ndarray()):
            for idx in ti.grouped(result):
                i, j, x, y, z = idx
                result[idx] = (rho_u[i, x, y, z] * B[j, x, y, z]) / rho[get_sc_idx(idx)]

        Lu = ti.ndarray(dtype=ti.float64, shape=(3, 3, ) + self.config.shape)
        Lu_filter = ti.ndarray(dtype=ti.float64, shape=get_filtered_shape(Lu.shape, FILTER_SIZE))


        rho_u_ = ti.ndarray(dtype=ti.f64, shape=self.config.v_shape)
        rho_u_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(self.config.v_shape, FILTER_SIZE))
        rho_u(rho, u, rho_u_)

        filter_vec(rho_u_, rho_u_filter, FILTER_SIZE)


        Lu_a_ = ti.ndarray(dtype=ti.f64, shape=Lu.shape)
        Lu_A(rho, rho_u_, Lu_a_)
        Lu_a_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(Lu.shape, FILTER_SIZE))

        filter_mat(Lu_a_, Lu_a_filter, FILTER_SIZE)

        Lu_a_ = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(Lu.shape, FILTER_SIZE))

        Lu_A(rho_filter, rho_u_filter, Lu_a_)

        BiBj_ = ti.ndarray(dtype=ti.f64, shape=Lu.shape)
        BiBj_filter = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(Lu.shape, FILTER_SIZE))
        BiBj(B, BiBj_)
        filter_mat(BiBj_, BiBj_filter, FILTER_SIZE)

        BiBj_ = ti.ndarray(dtype=ti.f64, shape=get_filtered_shape(Lu.shape, FILTER_SIZE))

        BiBj(B_filter, BiBj_)

        Ma = 1.1
        @ti.kernel
        def Lu_compute(Lu: ti.types.ndarray(),
                       BiBj_: ti.types.ndarray(),
                       BiBj_filter: ti.types.ndarray(),
                       Lu_A_: ti.types.ndarray(),
                       Lu_A_filter: ti.types.ndarray(),):
            print(Lu_A_filter.shape)
            for idx in ti.grouped(Lu):
                Lu[idx] = Lu_A_filter[idx] - Lu_A_[idx] - (BiBj_filter[idx] - BiBj_[idx])/Ma**2

        Lu_compute(Lu_filter, BiBj_, BiBj_filter, Lu_a_, Lu_a_filter)



    def _les_filter(self, arr: cla.Array, filter_size=2):
        filter_shape = tuple([filter_size, filter_size, filter_size])
        new_shape = tuple(s//filter_size for s in self.config.shape)
        Logger.log(new_shape)
        filtered_arr = cla.empty(self.queue, new_shape, dtype=arr.dtype)

        ary = cla.empty(self.queue, filter_shape, dtype=np.float64)
        dest_idx = cla.to_device(self.queue, ary=np.arange(0, prod(filter_shape), dtype=np.int32))
        for x in range(new_shape[0]):
            for y in range(new_shape[1]):
                for z in range(new_shape[2]):
                    # Logger.log(f"Map from [{x*filter_size}:{(x+1)*filter_size},\
                    #  {y*filter_size}:{(y+1)*filter_size}, \
                    # {z*filter_size}:{(z+1)*filter_size}] to [{x},{y},{z}]")
                    host_src_idx = self._get_list_idx(x0=x*filter_size, x1=(x+1)*filter_size, 
                                                      y0=y*filter_size, y1=(y+1)*filter_size, 
                                                      z0=z*filter_size, z1=(z+1)*filter_size)
                    src_idx = cla.to_device(queue=self.queue, ary=host_src_idx)
                    cla.multi_take_put(queue=self.queue, arrays=[arr], 
                                       src_indices=src_idx, dest_indices=dest_idx, out=[ary])
                    filtered_arr[x, y, z] = cla.sum(queue=self.queue, a=ary) / prod(filter_shape)
        return filtered_arr
    
    def _les_v_filter(self, arr: cla.Array, filter_size=2):
        filter_shape = tuple([filter_size, filter_size, filter_size])
        new_shape = (3, ) + tuple(s//filter_size for s in self.config.shape)
        Logger.log(new_shape)
        filtered_arr = cla.empty(self.queue, new_shape, dtype=arr.dtype)

        ary = cla.empty(self.queue, filter_shape, dtype=np.float64)
        dest_idx = cla.to_device(self.queue, ary=np.arange(0, prod(filter_shape), dtype=np.int32))
        for ax in range(3):
            for x in range(new_shape[0]):
                for y in range(new_shape[1]):
                    for z in range(new_shape[2]):
                        # Logger.log(f"Map from [{ax}, {x*filter_size}:{(x+1)*filter_size},\
                        #  {y*filter_size}:{(y+1)*filter_size}, \
                        # {z*filter_size}:{(z+1)*filter_size}] to [{ax}, {x},{y},{z}]")
                        host_src_idx = self._get_list_v_idx(ax, x0=x*filter_size, x1=(x+1)*filter_size, 
                                                        y0=y*filter_size, y1=(y+1)*filter_size, 
                                                        z0=z*filter_size, z1=(z+1)*filter_size)
                        src_idx = cla.to_device(queue=self.queue, ary=host_src_idx)
                        cla.multi_take_put(queue=self.queue, arrays=[arr], 
                                        src_indices=src_idx, dest_indices=dest_idx, out=[ary])
                        filtered_arr[ax, x, y, z] = cla.sum(queue=self.queue, a=ary) / prod(filter_shape)
        return filtered_arr

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
