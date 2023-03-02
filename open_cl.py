#!/usr/bin/env python

import os
import numpy as np
import pyopencl as cl
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

L = 2*np.pi
GHOSTS = config.GHOSTS

T_SHAPE = config.T_SHAPE

h = (L/T_SHAPE[0], L/T_SHAPE[1], L/T_SHAPE[2])

dV = h[0]*h[1]*h[2]
SHAPE = config.SHAPE
FLUX_SHAPE = (T_SHAPE[0]+1, T_SHAPE[1]+1, T_SHAPE[2]+1,)

START_STEP = config.START_STEP
steps = config.STEPS
T =  steps * 0.5*(1/T_SHAPE[0])**2
RW_DELETIMER = config.RW_DELETIMER

scalar_shape = SHAPE
vec_shape = (3,) + SHAPE

e_kin = np.zeros(1).astype(np.float64)
e_mag = np.zeros(1).astype(np.float64)

deltaT = np.zeros(1).astype(np.float64)

u = np.zeros(vec_shape).astype(np.float64)

B_arr = np.zeros(vec_shape).astype(np.float64)

p = np.zeros(scalar_shape).astype(np.float64)

rho = np.zeros(scalar_shape).astype(np.float64)

def evaluate_ghosts(_knl, queue,_rho, _p, _u, _B):
  evt = _knl(queue, (2*GHOSTS, SHAPE[1], SHAPE[2]), None, np.int32(0), _rho, _p, _u, _B)
  evt.wait()

  evt = _knl(queue, (SHAPE[0], 2*GHOSTS, SHAPE[2]), None, np.int32(1), _rho, _p, _u, _B)
  evt.wait()

  evt = _knl(queue, (SHAPE[0], SHAPE[1], 2*GHOSTS), None, np.int32(2), _rho, _p, _u, _B)
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

  data_service = DataService(str(date.today()) + "_LES_v1", scalar_shape, vec_shape, rw_energy=config.REWRITE_ENERGY)
  # data_service = DataService("2023-02-15" + "test", scalar_shape, vec_shape, rw_energy=config.REWRITE_ENERGY)

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

  dT_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=deltaT)

  p_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=p)
  rho_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
  u_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
  B_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B_arr)

  pk1_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=p)
  rk1_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
  uk1_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
  Bk1_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B_arr)

  pk2_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=p)
  rk2_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
  uk2_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
  Bk2_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B_arr)

  rho_flux_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
  u_flux_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
  B_flux_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B_arr)

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

  knl_int_kin = prg.integrate_kinetic
  knl_int_mag = prg.integrate_magnetic

  if start_step == 0:
    timing.add_timestamp('execute init program start')
    print('execute init program start\n')

    evt = knl_init_Orszag(queue, T_SHAPE, None, rho_buff, p_buff, u_buff, B_buff)
    evt.wait()

    # initials(rho, p, u, B_arr)

    # cl.enqueue_copy(queue, rho_buff, rho)
    # cl.enqueue_copy(queue, u_buff, u)
    # cl.enqueue_copy(queue, B_buff, B_arr)
    # cl.enqueue_copy(queue, p_buff, p)

    evaluate_ghosts(knl_ghosts, queue, rho_buff, p_buff, u_buff, B_buff)
    
    timing.add_timestamp('execute init program end')
    print('execute init program end\n')

    cl.enqueue_copy(queue, rho, rho_buff)
    cl.enqueue_copy(queue, u, u_buff)
    cl.enqueue_copy(queue, B_arr, B_buff)
    cl.enqueue_copy(queue, p, p_buff)

    timing.add_timestamp('compute_kin_energy start')
    print('compute_kin_energy start\n')
    # knl, ctx, mf, queue, rho_gpu, u_gpu
    e_kin[0] = compute_kin_energy(knl_int_kin, ctx, mf, queue, rho_buff, u_buff)
    timing.add_timestamp('compute_kin_energy end')
    print('compute_kin_energy end\n')

    timing.add_timestamp('compute_mag_energy start')
    print('compute_mag_energy start\n')
    e_mag[0] = compute_mag_energy()
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
    dT = 0.5*(1/T_SHAPE[0])**2

    t += dT
    if i == start_step and start_step != 0:
      timing.add_elapsed_time_start()
      print(f'Reading step_{i} file:')
      data_service.read_data(i, (u, B_arr, rho, p))

      cl.enqueue_copy(queue, rho_buff, rho)
      cl.enqueue_copy(queue, p_buff, p)
      cl.enqueue_copy(queue, u_buff, u)
      cl.enqueue_copy(queue, B_buff, B_arr)
      timing.add_elapsed_time_end("elapsed_reading_time")

    timing.add_elapsed_time_start()

    deltaT[0] = dT

    cl.enqueue_copy(queue, dT_buff, deltaT)

    # evt = knl_fluxes(queue, FLUX_SHAPE, None, 
    #   rho_buff, p_buff, u_buff, B_buff,
    #   rho_flux_buff, u_flux_buff, B_flux_buff,
    # )
    # evt.wait()

    evt = knl_solve_0(queue, T_SHAPE, None, 
      dT_buff, rho_buff, p_buff, u_buff, B_buff,
      rk1_buff, pk1_buff, uk1_buff, Bk1_buff,
      rho_flux_buff, u_flux_buff, B_flux_buff,
    )
    evt.wait()

    evaluate_ghosts(knl_ghosts, queue, rk1_buff, pk1_buff, uk1_buff, Bk1_buff)

    evt = knl_solve_1(queue, T_SHAPE, None, 
      dT_buff, rho_buff, p_buff, u_buff, B_buff,
      rk1_buff, pk1_buff, uk1_buff, Bk1_buff,
      rk2_buff, pk2_buff, uk2_buff, Bk2_buff,
      rho_flux_buff, u_flux_buff, B_flux_buff,
    )
    evt.wait()

    evaluate_ghosts(knl_ghosts, queue, rk2_buff, pk2_buff, uk2_buff, Bk2_buff)

    # evt = knl_fluxes(queue, FLUX_SHAPE, None, 
    #   rk2_buff, pk2_buff, uk2_buff, Bk2_buff,
    #   rho_flux_buff, u_flux_buff, B_flux_buff,
    # )
    # evt.wait()

    evt = knl_solve_2(queue, T_SHAPE, None, 
      dT_buff, rho_buff, p_buff, u_buff, B_buff,
      rk1_buff, pk1_buff, uk1_buff, Bk1_buff,
      rk2_buff, pk2_buff, uk2_buff, Bk2_buff,
      rho_flux_buff, u_flux_buff, B_flux_buff,
    )
    evt.wait()

    evaluate_ghosts(knl_ghosts, queue, rk1_buff, pk1_buff, uk1_buff, Bk1_buff)

    cl.enqueue_copy(queue, rho_buff, rk1_buff)
    cl.enqueue_copy(queue, u_buff, uk1_buff)
    cl.enqueue_copy(queue, B_buff, Bk1_buff)
    cl.enqueue_copy(queue, p_buff, pk1_buff)

    timing.add_elapsed_time_end("elapsed_kernel_time")

    if (i+1) % RW_DELETIMER == 0:
      timing.add_elapsed_time_start()
      print(f"Step: {i}, t: {t}")
      print(f'Writing step_{i+1} file:')
      cl.enqueue_copy(queue, rho, rho_buff)
      cl.enqueue_copy(queue, u, u_buff)
      cl.enqueue_copy(queue, B_arr, B_buff)
      cl.enqueue_copy(queue, p, p_buff)
      e_kin = np.append(e_kin, compute_kin_energy(knl_int_kin, ctx, mf, queue, rho_buff, u_buff))
      e_mag = np.append(e_mag, compute_mag_energy())
      data_service.save_data(i+1, (u, B_arr, rho, p))
      timing.add_elapsed_time_end("elapsed_writing_time")
    i += 1

  if (i) % RW_DELETIMER != 0:
    print(f'Writing step_{i+1} file:')
    cl.enqueue_copy(queue, rho, rho_buff)
    cl.enqueue_copy(queue, u, u_buff)
    cl.enqueue_copy(queue, B_arr, B_buff)
    cl.enqueue_copy(queue, p, p_buff)
    e_kin = np.append(e_kin, compute_kin_energy(knl_int_kin, ctx, mf, queue, rho_buff, u_buff))
    e_mag = np.append(e_mag, compute_mag_energy())
    data_service.save_data(i+1, (u, B_arr, rho, p))

  data_service.save_energy((e_kin, e_mag))

  with  open(data_service.get_or_create_dir('timig') / 'timig.txt', 'w') as f:
    print(timing.time_stamps, file=f)
    print(timing.elapsed_times, file=f)


def compute_kin_energy(knl, ctx, mf, queue, rho_gpu, u_gpu):
  result = 0.0
  for i in range(0, 3):
    for x in range(GHOSTS, SHAPE[2] - GHOSTS):
      for y in range(GHOSTS, SHAPE[2] - GHOSTS):
        for z in range(GHOSTS, SHAPE[2] - GHOSTS):
          result += dV * rho[x, y, z] * u[i, x, y, z]**2
  # return result
  start_local_shape = (64, 64, 64,)
  global_shape = SHAPE
  global_size = SHAPE[0]*SHAPE[1]*SHAPE[2]
  local_shape = start_local_shape

  sums = np.zeros((SHAPE[0] // local_shape[0], 
                  SHAPE[1] // local_shape[1], 
                  SHAPE[2] // local_shape[2])).astype(np.float64)
  
  local_size = local_shape[0]*local_shape[1]*local_shape[2] * 8

  while global_size//local_size > 1:
    local_size = local_shape[0]*local_shape[1]*local_shape[2] * 8
    localSums = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, size=local_size)
    partialSums = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, size=global_size//local_size)

    evt = knl(queue, global_shape, None, np.int32(0), 
              rho_gpu, u_gpu, partialSums, localSums)
    
    global_shape = (global_shape[0]//local_shape[0], 
                    global_shape[1]//local_shape[1], 
                    global_shape[2]//local_shape[2])
    
    global_size = global_size//local_size
    evt.wait()

    
    cl.enqueue_copy(queue, sums, partialSums)

  print(result, sums[0, 0, 0])
  assert(np.allclose([result, ], [sums[0, 0, 0], ]))
  return sums[0, 0, 0]





def compute_mag_energy():
  result = 0.0
  for i in range(0, 3):
    for x in range(GHOSTS, SHAPE[2] - GHOSTS):
      for y in range(GHOSTS, SHAPE[2] - GHOSTS):
        for z in range(GHOSTS, SHAPE[2] - GHOSTS):
          result += dV * B_arr[i, x, y, z]**2
  return result




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
