#!/usr/bin/env python

import os
import numpy as np

import pyopencl as cl
import pyopencl.array

from pathlib import Path
from timing import Timing
from datetime import date
import matplotlib.pyplot as plt

from config import Config
from data_service import DataService

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_NO_CACHE'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

config = Config()

GHOSTS = config.GHOSTS

T_SHAPE = config.T_SHAPE
SHAPE = config.SHAPE
FLUX_SHAPE = (T_SHAPE[0]+1, T_SHAPE[1]+1, T_SHAPE[2]+1,)

START_STEP = config.START_STEP
steps = config.STEPS
T =  steps * 0.5*(1/T_SHAPE[0])**2
RW_DELETIMER = config.RW_DELETIMER

L = 2*np.pi
h = (L/T_SHAPE[0], L/T_SHAPE[1], L/T_SHAPE[2])
dV = h[0]*h[1]*h[2]

scalar_shape = SHAPE
vec_shape = (3,) + SHAPE

t_scalar_shape = T_SHAPE
t_vec_shape = (3,) + T_SHAPE

e_kin = np.zeros(1).astype(np.float64)
e_mag = np.zeros(1).astype(np.float64)

deltaT = np.zeros(1).astype(np.float64)

u = np.zeros(vec_shape).astype(np.float64)

B_arr = np.zeros(vec_shape).astype(np.float64)

p = np.zeros(scalar_shape).astype(np.float64)

rho = np.zeros(scalar_shape).astype(np.float64)

def evaluate_ghosts(_knl, queue,_rho, _p, _u, _B):
  evt = _knl(queue, (2*GHOSTS, SHAPE[1], SHAPE[2]), None, np.int32(0), _rho.data, _p.data, _u.data, _B.data)
  evt.wait()

  evt = _knl(queue, (SHAPE[0], 2*GHOSTS, SHAPE[2]), None, np.int32(1), _rho.data, _p.data, _u.data, _B.data)
  evt.wait()

  evt = _knl(queue, (SHAPE[0], SHAPE[1], 2*GHOSTS), None, np.int32(2), _rho.data, _p.data, _u.data, _B.data)
  evt.wait()

def initials(rho, p, u, B):
  B0 = 0.282094
  gamma = 5.0/3.0
  p0 = 5.0/(12.0*np.pi)
  rho0 = 25.0/(36.0*np.pi)
  Ms = 0.01
  Cs = 1.0
  u0 = (Ms * Cs)

  for x in range(GHOSTS, T_SHAPE[0] + GHOSTS):
    for y in range(GHOSTS, T_SHAPE[1] + GHOSTS):
      for z in range(GHOSTS, T_SHAPE[2] + GHOSTS):
        p[x][y][z] = p0 + 1e-3*np.random.rand()
        rho[x][y][z] = rho0 + 1e-3*np.random.rand()

        u[0][x][y][z] = u0 + 1e-3*np.random.rand()
        u[1][x][y][z] = u0 + 1e-3*np.random.rand()
        u[2][x][y][z] = u0 + 1e-3*np.random.rand()

        B[0][x][y][z] = B0 + 1e-3*np.random.rand()
        B[1][x][y][z] = B0 + 1e-3*np.random.rand()
        B[2][x][y][z] = B0 + 1e-3*np.random.rand()

def main(start_step = 0):
  global u, B_arr, rho, e_kin, e_mag

  # data_service = DataService(str(date.today()) + "test", scalar_shape, vec_shape, rw_energy=config.REWRITE_ENERGY)
  data_service = DataService("2023-03-02_DNS", scalar_shape, vec_shape, rw_energy=config.REWRITE_ENERGY)

  timing = Timing()

  timing.add_timestamp("start")
  print('load program from cl source file')
  with open(Path('./c_sources/main.cl'), 'r') as file:
    data = file.read()
    
  print('create context ...')
  ctx = cl.create_some_context()

  print('create command queue ...')
  queue = cl.CommandQueue(ctx)

  mf = cl.mem_flags

  timing.add_timestamp('prepare device memory for input / output')
  print('prepare device memory for input / output')

  kin_energy_gpu = cl.array.empty(queue, T_SHAPE, dtype=np.float64)
  mag_energy_gpu = cl.array.empty(queue, T_SHAPE, dtype=np.float64)

  p_gpu = cl.array.empty(queue, p.shape, dtype=np.float64)
  rho_gpu = cl.array.empty(queue, rho.shape, dtype=np.float64)
  u_gpu = cl.array.empty(queue, u.shape, dtype=np.float64)
  B_gpu = cl.array.empty(queue, B_arr.shape, dtype=np.float64)

  pk1_gpu = cl.array.empty(queue, p.shape, dtype=np.float64)
  rk1_gpu = cl.array.empty(queue, rho.shape, dtype=np.float64)
  uk1_gpu = cl.array.empty(queue, u.shape, dtype=np.float64)
  Bk1_gpu = cl.array.empty(queue, B_arr.shape, dtype=np.float64)

  pk2_gpu = cl.array.empty(queue, p.shape, dtype=np.float64)
  rk2_gpu = cl.array.empty(queue, rho.shape, dtype=np.float64)
  uk2_gpu = cl.array.empty(queue, u.shape, dtype=np.float64)
  Bk2_gpu = cl.array.empty(queue, B_arr.shape, dtype=np.float64)

  rho_flux_gpu = cl.array.empty(queue, rho.shape, dtype=np.float64)
  u_flux_gpu = cl.array.empty(queue, u.shape, dtype=np.float64)
  B_flux_gpu = cl.array.empty(queue, B_arr.shape, dtype=np.float64)

  timing.add_timestamp('compile kernel code')
  print('compile kernel code')
  prg = cl.Program(ctx, data).build(options=['-I', './c_sources'])

  knl_init_Orszag = prg.Orszag_Tang_3D_inital
  # knl_init_tanh = prg.Tanh_3D_inital

  knl_solve_0 = prg.solver_3D_RK0
  knl_solve_1 = prg.solver_3D_RK1
  knl_solve_2 = prg.solver_3D_RK2

  knl_fluxes = prg.compute_fluxes_3D
  knl_ghosts = prg.ghost_nodes_periodic

  knl_kin_e = prg.kin_energy
  knl_mag_e = prg.mag_energy

  if start_step == 0:
    timing.add_timestamp('execute init program start')
    print('execute init program start\n')

    evt = knl_init_Orszag(queue, T_SHAPE, None, rho_gpu.data, p_gpu.data, u_gpu.data, B_gpu.data)
    evt.wait()

    # initials(rho, p, u, B_arr)

    # cl.enqueue_copy(queue, rho_gpu, rho)
    # cl.enqueue_copy(queue, u_gpu, u)
    # cl.enqueue_copy(queue, B_gpu, B_arr)
    # cl.enqueue_copy(queue, p_gpu, p)

    evaluate_ghosts(knl_ghosts, queue, rho_gpu, p_gpu, u_gpu, B_gpu)
    
    timing.add_timestamp('execute init program end')
    print('execute init program end\n')

    rho_gpu.get(ary=rho)
    u_gpu.get(ary=u)
    B_gpu.get(ary=B_arr)
    p_gpu.get(ary=p)

    timing.add_timestamp('compute_kin_energy start')
    print('compute_kin_energy start\n')
    e_kin[0] = compute_kin_energy(knl_kin_e, queue, rho_gpu, u_gpu, kin_energy_gpu)
    timing.add_timestamp('compute_kin_energy end')
    print('compute_kin_energy end\n')

    timing.add_timestamp('compute_mag_energy start')
    print('compute_mag_energy start\n')
    e_mag[0] = compute_mag_energy(knl_mag_e, queue, B_gpu, mag_energy_gpu)
    timing.add_timestamp('compute_mag_energy end')
    print('compute_mag_energy end\n')

    data_service.save_data(0, (u, B_arr, rho, p))

  else:
    e_kin = np.delete(e_kin,-1)
    e_mag = np.delete(e_mag,-1)

  timing.add_timestamp('execute kernel programs')
  print('execute kernel programs\n')

  t = 0

  global T
  global steps
  i = start_step
  while t < T:
    dT = np.float64(0.5*(1/T_SHAPE[0])**2)

    t += dT
    if i == start_step and start_step != 0:
      timing.add_elapsed_time_start()
      print(f'Reading step_{i} file:')
      data_service.read_data(i, (u, B_arr, rho, p))
      
      rho_gpu.set(ary=rho)
      u_gpu.set(ary=u)
      B_gpu.set(ary=B_arr)
      p_gpu.set(ary=p)
      timing.add_elapsed_time_end("elapsed_reading_time")

    timing.add_elapsed_time_start()

    # evt = knl_fluxes(queue, FLUX_SHAPE, None, 
    #   rho_gpu, p_gpu, u_gpu, B_gpu,
    #   rho_flux_gpu, u_flux_gpu, B_flux_gpu,
    # )
    # evt.wait()

    evt = knl_solve_0(queue, T_SHAPE, None, 
      np.float64(dT), rho_gpu.data, p_gpu.data, u_gpu.data, B_gpu.data,
      rk1_gpu.data, pk1_gpu.data, uk1_gpu.data, Bk1_gpu.data,
      rho_flux_gpu.data, u_flux_gpu.data, B_flux_gpu.data,
    )
    evt.wait()

    evaluate_ghosts(knl_ghosts, queue, rk1_gpu, pk1_gpu, uk1_gpu, Bk1_gpu)

    evt = knl_solve_1(queue, T_SHAPE, None, 
      np.float64(dT), rho_gpu.data, p_gpu.data, u_gpu.data, B_gpu.data,
      rk1_gpu.data, pk1_gpu.data, uk1_gpu.data, Bk1_gpu.data,
      rk2_gpu.data, pk2_gpu.data, uk2_gpu.data, Bk2_gpu.data,
      rho_flux_gpu.data, u_flux_gpu.data, B_flux_gpu.data,
    )
    evt.wait()

    evaluate_ghosts(knl_ghosts, queue, rk2_gpu, pk2_gpu, uk2_gpu, Bk2_gpu)

    # evt = knl_fluxes(queue, FLUX_SHAPE, None, 
    #   rk2_gpu, pk2_gpu, uk2_gpu, Bk2_gpu,
    #   rho_flux_gpu, u_flux_gpu, B_flux_gpu,
    # )
    # evt.wait()

    evt = knl_solve_2(queue, T_SHAPE, None, 
      np.float64(dT), rho_gpu.data, p_gpu.data, u_gpu.data, B_gpu.data,
      rk1_gpu.data, pk1_gpu.data, uk1_gpu.data, Bk1_gpu.data,
      rk2_gpu.data, pk2_gpu.data, uk2_gpu.data, Bk2_gpu.data,
      rho_flux_gpu.data, u_flux_gpu.data, B_flux_gpu.data,
    )
    evt.wait()

    evaluate_ghosts(knl_ghosts, queue, rk1_gpu, pk1_gpu, uk1_gpu, Bk1_gpu)

    rho_gpu = rk1_gpu
    u_gpu = uk1_gpu
    B_gpu = Bk1_gpu
    p_gpu = pk1_gpu

    timing.add_elapsed_time_end("elapsed_kernel_time")

    if (i+1) % RW_DELETIMER == 0:
      timing.add_elapsed_time_start()
      print(f"Step: {i}, t: {t}")
      print(f'Writing step_{i+1} file:')

      rho_gpu.get(ary=rho)
      u_gpu.get(ary=u)
      B_gpu.get(ary=B_arr)
      p_gpu.get(ary=p)

      e_kin = np.append(e_kin, compute_kin_energy(knl_kin_e, queue, rho_gpu, u_gpu, kin_energy_gpu))
      e_mag = np.append(e_mag, compute_mag_energy(knl_mag_e, queue, B_gpu, mag_energy_gpu))
      data_service.save_data(i+1, (u, B_arr, rho, p))
      timing.add_elapsed_time_end("elapsed_writing_time")
    i += 1

  if (i) % RW_DELETIMER != 0:
    print(f'Writing step_{i+1} file:')

    rho_gpu.get(ary=rho)
    u_gpu.get(ary=u)
    B_gpu.get(ary=B_arr)
    p_gpu.get(ary=p)
    
    e_kin = np.append(e_kin, compute_kin_energy(knl_kin_e, queue, rho_gpu, u_gpu, kin_energy_gpu))
    e_mag = np.append(e_mag, compute_mag_energy(knl_mag_e, queue, B_gpu, mag_energy_gpu))
    data_service.save_data(i+1, (u, B_arr, rho, p))

  data_service.save_energy((e_kin, e_mag))

  with  open(data_service.get_or_create_dir('timig') / 'timig.txt', 'w') as f:
    print(timing.time_stamps, file=f)
    print(timing.elapsed_times, file=f)


def compute_kin_energy(knl_kin_e, queue, rho_gpu, u_gpu, kin_energy_gpu):
  # result = 0.0
  # for i in range(0, 3):
  #   for x in range(GHOSTS, SHAPE[2] - GHOSTS):
  #     for y in range(GHOSTS, SHAPE[2] - GHOSTS):
  #       for z in range(GHOSTS, SHAPE[2] - GHOSTS):
  #         result += 0.5 * dV * rho[x, y, z] * u[i, x, y, z]**2

  evt = knl_kin_e(queue, T_SHAPE, None, rho_gpu.data, u_gpu.data, kin_energy_gpu.data)
  evt.wait()

  energy = cl.array.sum(kin_energy_gpu)
  return energy.get()


def compute_mag_energy(knl_mag_e, queue, B_gpu, mag_energy_gpu):
  # result = 0.0
  # for i in range(0, 3):
  #   for x in range(GHOSTS, SHAPE[2] - GHOSTS):
  #     for y in range(GHOSTS, SHAPE[2] - GHOSTS):
  #       for z in range(GHOSTS, SHAPE[2] - GHOSTS):
  #         result += 0.5 * dV * B_arr[i, x, y, z]**2

  evt = knl_mag_e(queue, T_SHAPE, None, B_gpu.data, mag_energy_gpu.data)
  evt.wait()

  energy = cl.array.sum(mag_energy_gpu)
  return energy.get()


def plot_energy(end_step):
  global u, B_arr, rho, p, e_kin, e_mag
  data_service = DataService(str(date.today()) + 'v2', scalar_shape, vec_shape)

  e_kin = np.array([])
  e_mag = np.array([])
  for i in range(0, end_step, RW_DELETIMER):
    print(f"Step: {i}")
    data_service.read_data(i, (u, B_arr, rho, p))

    e_kin = np.append(e_kin, compute_kin_energy())
    e_mag = np.append(e_mag, compute_mag_energy())
  
  data_service.save_energy((e_kin, e_mag))


if __name__ == "__main__":
    main(start_step=START_STEP)
