#!/usr/bin/env python

import os
import h5py
import time
import numpy as np
import pyopencl as cl
from pathlib import Path
from datetime import date
import matplotlib.pyplot as plt

from data_service import DataService

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_NO_CACHE'] = '1'

work_dir = '13_10_22'#'05_12_22'

SHAPE = (256, 256, 256)
STEPS = 100
RW_DELETIMER = 10

scalar_shape = SHAPE
vec_shape = (3,) + SHAPE

rho = np.zeros(np.product(scalar_shape)).astype(np.float32)
u = np.zeros(np.product(vec_shape)).astype(np.float32)
B = np.zeros(np.product(vec_shape)).astype(np.float32)

def read_values(step):
  global rho 
  global B 
  global u 

  with h5py.File(Path(f"./{work_dir}/data/step_{step}.hdf5"), "r") as f:
    u.reshape(vec_shape)[:] = f['u']
    B.reshape(vec_shape)[:] = f['B']
    rho.reshape(scalar_shape)[:] = f['ro']


def write_values(step):
  with h5py.File(Path(f"./{work_dir}/data/step_{step}.hdf5"), "w") as f:
    dset = f.create_dataset("u", vec_shape, dtype=np.float32)
    dset[:] = u.reshape(vec_shape)
    dset = f.create_dataset("B", vec_shape, dtype=np.float32)
    dset[:] = B.reshape(vec_shape)
    dset = f.create_dataset("ro", scalar_shape, dtype=np.float32)
    dset[:] = rho.reshape(scalar_shape)


def main(start_step = 0):
  global u, B, rho
  data_service = DataService(date.today(), scalar_shape, vec_shape)

  print('load program from cl source file')
  with open(Path('./c_sources/source.cl'), 'r') as file:
    data = file.read()
    
  print('create context ...')
  platforms = cl.get_platforms()
  print('Platforms: ', platforms)
  input()
  ctx = cl.Context(
    dev_type=cl.device_type.ALL,
    properties=[(cl.context_properties.PLATFORM, platforms[0])])

  print('create command queue ...')
  queue = cl.CommandQueue(ctx)

  mf = cl.mem_flags

  print('prepare device memory for input / output')

  rho_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
  u_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
  B_buff = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B)

  new_rho = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho)
  new_u = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
  new_B = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B)


  print('compile kernel code')
  prg = cl.Program(ctx, data).build(options=['-I', './c_sources'])

  knl_init = prg.orzang_tang
  knl_solve = prg.solve_system

# if start_step == 0:
  evt = knl_init(queue, scalar_shape, None, u_buff, B_buff, rho_buff)
  evt.wait()

  cl.enqueue_copy(queue, rho, rho_buff)
  cl.enqueue_copy(queue, u, u_buff)
  cl.enqueue_copy(queue, B, B_buff)

  data_service.save_data(0, (u, B, rho))

  print('execute kernel programs\n')

  for i in range(start_step, start_step + STEPS):
    print(f"Step: {i}")
    if i == start_step:
      print(f'Reading step_{i} file:')
      data_service.read_data(i, (u, B, rho))

      cl.enqueue_copy(queue, rho_buff, rho)
      cl.enqueue_copy(queue, u_buff, u)
      cl.enqueue_copy(queue, B_buff, B)

    evt = knl_solve(queue, scalar_shape, None, 
      u_buff, B_buff, rho_buff, new_u, new_B, new_rho)
    evt.wait()

    cl.enqueue_copy(queue, rho_buff, new_rho)
    cl.enqueue_copy(queue, u_buff, new_u)
    cl.enqueue_copy(queue, B_buff, new_B)

    if (i+1) % RW_DELETIMER == 0:
      print(f'Writing step_{i+1} file:')
      cl.enqueue_copy(queue, rho, new_rho)
      cl.enqueue_copy(queue, u, new_u)
      cl.enqueue_copy(queue, B, new_B)
      data_service.save_data(i+1, (u, B, rho))


def compute_kinetic_energy(start_step, end_step):
    E = np.zeros((end_step-start_step)//RW_DELETIMER)
    it = 0
    x = list(range(start_step//RW_DELETIMER,end_step//RW_DELETIMER))
    for i in range(len(E)):
        read_values(x[i]*RW_DELETIMER)
        E[i] = 0.5 * np.sum(
          [u.reshape(vec_shape)[i, :]**2/rho.reshape(scalar_shape)[:] for i in range(3)]
          )
    fig = plt.figure()
    ax = plt.axes()
    X = [i for i in x]
    print(E)
    print(x)
    plt.grid()
    plt.xlabel('Шаг по времени')
    plt.ylabel('Суммарная кин. энергия')
    plt.plot(X, E)
    plt.savefig(Path(f"./{work_dir}/graphs/kin_energy_{start_step}_to_{end_step}.jpg"))

def compute_magnetic_energy(start_step, end_step):
    E = np.zeros((end_step-start_step)//RW_DELETIMER)
    it = 0
    x = list(range(start_step//RW_DELETIMER,end_step//RW_DELETIMER))
    for i in x:
        read_values(i*RW_DELETIMER)
        E[i] = 0.5 * np.sum([j**2 for j in B.reshape(vec_shape)[0, :]])
    fig = plt.figure()
    ax = plt.axes()
    X = [i for i in x]
    plt.plot(X, E)
    plt.savefig(Path(f"./{work_dir}/graphs/mag_energy_{start_step}_to_{end_step}.jpg"))

def sum_ro(start_step, end_step):
    E = np.zeros((end_step-start_step)//RW_DELETIMER)
    it = 0
    x = list(range(start_step//RW_DELETIMER,end_step//RW_DELETIMER))
    for i in range(len(E)):
        read_values(x[i]*RW_DELETIMER)
        E[i] = np.sum([j for j in rho.reshape(scalar_shape)[0, :]])
    fig = plt.figure()
    ax = plt.axes()
    X = [i for i in x]
    plt.grid()
    plt.xlabel('Шаг по времени')
    plt.ylabel('Суммарная масса системы')
    plt.plot(X, E)
    plt.savefig(Path(f"./{work_dir}/graphs/ro_sum_{start_step}_to_{end_step}.jpg"))
  
def sum_quad_u(start_step, end_step):
    E = np.zeros((end_step-start_step)//RW_DELETIMER)
    it = 0
    x = list(range(start_step//RW_DELETIMER,end_step//RW_DELETIMER))
    for i in x:
        read_values(i*RW_DELETIMER)
        E[i] = np.sum([j**2 for j in u.reshape(vec_shape)[0, :]])
    fig = plt.figure()
    ax = plt.axes()
    X = [i*DELTA_TAU for i in x]
    plt.plot(X, E)
    plt.savefig(Path(f"./{work_dir}/graphs/sum_quad_u.jpg"))


def test():

  import os
  os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
  os.environ['PYOPENCL_NO_CACHE'] = '1'

  DELTA_H = (2*3.14)/256

  foo_g = np.zeros(SHAPE).astype(np.float32)
  div = np.zeros(SHAPE).astype(np.float32)
  res = np.zeros(SHAPE).astype(np.float32)

  print('load program from cl source file')
  with open(Path('./c_sources/my_math.cl'), 'r') as file:
    data = file.read()
    
  print('create context ...')
  platforms = cl.get_platforms()
  ctx = cl.Context(
    dev_type=cl.device_type.ALL,
    properties=[(cl.context_properties.PLATFORM, platforms[1])])

  print('create command queue ...')
  queue = cl.CommandQueue(ctx)

  mf = cl.mem_flags

  print('prepare device memory for input / output')

  foo = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=foo_g)
  res_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)


  print('compile kernel code')
  prg = cl.Program(ctx, data).build(options=['-I', './c_sources'])

  knl = prg.test_inital_data
  print('prepare init data ... ')

  evt = knl(queue, SHAPE, None, foo, res_g)
  evt.wait()

  cl.enqueue_copy(queue, foo_g, foo)
  cl.enqueue_copy(queue, div, res_g)

  knl = prg.test_second_der_x

  print('prepare data ... ')

  cl.enqueue_copy(queue, foo, foo_g)

  evt = knl(queue, SHAPE, None, foo, res_g)
  evt.wait()

  cl.enqueue_copy(queue, res, res_g)

  fig = plt.figure()
  ax = plt.axes()
  x = [i for i in range(SHAPE[2])]
  plt.plot(x, foo_g[:, 0, 0])
  plt.plot(x, res[:, 0, 0])
  plt.savefig(Path(f"./{work_dir}/graphs/test_d2x.jpg"))

  knl = prg.test_first_der_x

  print('prepare data ... ')

  cl.enqueue_copy(queue, foo, foo_g)

  evt = knl(queue, SHAPE, None, foo, res_g)
  evt.wait()

  cl.enqueue_copy(queue, res, res_g)

  fig = plt.figure()
  ax = plt.axes()
  x = [i for i in range(SHAPE[2])]
  plt.plot(x, foo_g[:, 0, 0])
  plt.plot(x, res[:, 0, 0])
  plt.savefig(Path(f"./{work_dir}/graphs/test_dx.jpg"))

  knl = prg.test_dxdy

  print('prepare data ... ')

  cl.enqueue_copy(queue, foo, foo_g)

  evt = knl(queue, SHAPE, None, foo, res_g)
  evt.wait()

  cl.enqueue_copy(queue, res, res_g)

  fig = plt.figure()
  ax = plt.axes()
  x = [i for i in range(SHAPE[2])]
  plt.plot(x, foo_g[:, 0, 0])
  plt.plot(x, res[:, 0, 0])
  plt.plot(x, div[:, 0, 0])
  plt.savefig(Path(f"./{work_dir}/graphs/test_dxdy.jpg"))

  knl = prg.test_dxdz

  print('prepare data ... ')

  cl.enqueue_copy(queue, foo, foo_g)

  evt = knl(queue, SHAPE, None, foo, res_g)
  evt.wait()

  cl.enqueue_copy(queue, res, res_g)

  fig = plt.figure()
  ax = plt.axes()
  x = [i for i in range(SHAPE[2])]
  plt.plot(x, foo_g[:, 0, 0])
  plt.plot(x, res[:, 0, 0])
  plt.plot(x, div[:, 0, 0])
  plt.savefig(Path(f"./{work_dir}/graphs/test_dxdz.jpg"))

  knl = prg.test_dydz

  print('prepare data ... ')

  cl.enqueue_copy(queue, foo, foo_g)

  evt = knl(queue, SHAPE, None, foo, res_g)
  evt.wait()

  cl.enqueue_copy(queue, res, res_g)

  fig = plt.figure()
  ax = plt.axes()
  x = [i for i in range(SHAPE[2])]
  plt.plot(x, foo_g[0, 0, :])
  plt.plot(x, res[0, 0, :])
  plt.plot(x, div[0, 0, :])
  plt.savefig(Path(f"./{work_dir}/graphs/test_dydz.jpg"))

def printv(start_step, end_step):
    it = 0
    x = list(range(0, 256))
    for i in x:
      read_values(i*RW_DELETIMER)
      fig = plt.figure()
      ax = plt.axes()
      plt.plot(x, rho.reshape(scalar_shape)[:, 0, 0])
      print(rho.reshape(scalar_shape)[:, 0, 0])
      plt.savefig(Path(f"./{work_dir}/graphs/u_x(0, 0)_{i}.jpg"))


if __name__ == "__main__":
    main()