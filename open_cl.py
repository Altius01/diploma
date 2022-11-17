#!/usr/bin/env python

import h5py
import time
import numpy as np
import pyopencl as cl
from pathlib import Path
import matplotlib.pyplot as plt

start_time = time.time()

work_dir = '13_10_22'

DELTA_TAU = 1e-5
DELTA_H = 1e-2


B_0 = 1

SHAPE = (256, 256, 256)
STEPS = 500
RW_DELETIMER = 10

scalar_shape = (2, ) + SHAPE
vec_shape = (2, 3,) + SHAPE
matr_shape = (2, 3, 3,) + SHAPE

ro = np.zeros(np.product(scalar_shape)).astype(np.float32)
u = np.zeros(np.product(vec_shape)).astype(np.float32)
B = np.zeros(np.product(vec_shape)).astype(np.float32)

f_0 = np.zeros(np.product(vec_shape)).astype(np.float32)
f_1 = np.zeros(np.product(matr_shape)).astype(np.float32)
f_2 = np.zeros(np.product(matr_shape)).astype(np.float32)

time_hostdata_loaded = time.time()

def read_values(step):
  global ro 
  global B 
  global u 
  global f_0 
  global f_1
  global f_2

  # ro = np.zeros(np.product(scalar_shape)).astype(np.float32)
  # u = np.zeros(np.product(vec_shape)).astype(np.float32)
  # B = np.zeros(np.product(vec_shape)).astype(np.float32)

  # f_0 = np.zeros(np.product(vec_shape)).astype(np.float32)
  # f_1 = np.zeros(np.product(matr_shape)).astype(np.float32)
  # f_2 = np.zeros(np.product(matr_shape)).astype(np.float32) 

  with h5py.File(Path(f"./{work_dir}/step_{step}.hdf5"), "r") as f:
    u.reshape(vec_shape)[0, :] = f['u']
    B.reshape(vec_shape)[0, :] = f['B']
    ro.reshape(scalar_shape)[0, :] = f['ro']
    f_0.reshape(vec_shape)[0, :] = f['f_0']
    f_1.reshape(matr_shape)[0, :] = f['f_1']
    f_2.reshape(matr_shape)[0, :] = f['f_2']


def write_values(step):
  with h5py.File(Path(f"./{work_dir}/step_{step}.hdf5"), "w") as f:
    dset = f.create_dataset("u", vec_shape[1:], dtype=np.float32)
    dset[:] = u.reshape(vec_shape)[1, :]

    dset = f.create_dataset("B", vec_shape[1:], dtype=np.float32)
    dset[:] = B.reshape(vec_shape)[1, :]

    dset = f.create_dataset("ro", scalar_shape[1:], dtype=np.float32)
    dset[:] = ro.reshape(scalar_shape)[1, :]

    dset = f.create_dataset("f_0", vec_shape[1:], dtype=np.float32)
    dset[:] = f_0.reshape(vec_shape)[1, :]

    dset = f.create_dataset("f_1", matr_shape[1:], dtype=np.float32)
    dset[:] = f_1.reshape(matr_shape)[1, :]

    dset = f.create_dataset("f_2", matr_shape[1:], dtype=np.float32)
    dset[:] = f_2.reshape(matr_shape)[1, :]

  with open(Path(f"./{work_dir}/u/u_{step}.txt"), "w") as f:
    f.write('u_x:\n')
    f.write(','.join([str(i) for i in u.reshape(vec_shape)[1, 0, :]]))
    f.write('\n\nu_y:\n')
    f.write(','.join([str(i) for i in u.reshape(vec_shape)[1, 1, :]]))
    f.write('\n\nu_z\n')
    f.write(','.join([str(i) for i in u.reshape(vec_shape)[1, 2, :]]))

  # with open(Path(f"./{work_dir}/f_0/f_0_{step}.txt"), "w") as f:
  #   f.write('f_0_x:\n')
  #   f.write(','.join([str(i) for i in f_0.reshape(vec_shape)[1, 0, :]]))
  #   f.write('\n\nf_0_y:\n')
  #   f.write(','.join([str(i) for i in f_0.reshape(vec_shape)[1, 1, :]]))
  #   f.write('\n\nf_0_z\n')
  #   f.write(','.join([str(i) for i in f_0.reshape(vec_shape)[1, 2, :]]))

  with open(Path(f"./{work_dir}/B/B_{step}.txt"), "w") as f:
    f.write('B_x:\n')
    f.write(','.join([str(i) for i in B.reshape(vec_shape)[1, 0, :]]))
    f.write('\n\nB_y:\n')
    f.write(','.join([str(i) for i in B.reshape(vec_shape)[1, 1, :]]))
    f.write('\n\nB_z\n')
    f.write(','.join([str(i) for i in B.reshape(vec_shape)[1, 2, :]]))

  with open(Path(f"./{work_dir}/ro/ro_{step}.txt"), "w") as f:
    f.write('ro:\n')
    f.write(','.join([str(i) for i in ro.reshape(scalar_shape)[1, :]]))


def initial():
  global ro 
  global B 
  global u 
  for t in range(2):
    for x in range(SHAPE[0]):
      for y in range(SHAPE[1]):
        for z in range(SHAPE[2]):
          X = tuple([x, y, z])
          ro.reshape(scalar_shape)[(t,)+X] = 25/(36*np.pi)
          B.reshape(vec_shape)[(t, 0) + X] = -B_0*np.sin(2*np.pi*y*DELTA_H)
          B.reshape(vec_shape)[(t, 1,) + X] = B_0*np.sin(2*np.pi*x*DELTA_H)

          u.reshape(vec_shape)[(t, 0,) + X] = -np.sin(2*np.pi*y*DELTA_H)
          u.reshape(vec_shape)[(t, 1,) + X] = np.sin(2*np.pi*x*DELTA_H)


def main():
  import os
  os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
  os.environ['PYOPENCL_NO_CACHE'] = '1'

  print('load program from cl source file')
  with open(Path('./c_sources/source.cl'), 'r') as file:
    data = file.read()
    
  print('create context ...')
  platforms = cl.get_platforms()
  print('Platforms: ', platforms)
  ctx = cl.Context(
    dev_type=cl.device_type.ALL,
    properties=[(cl.context_properties.PLATFORM, platforms[0])])

  print('create command queue ...')
  queue = cl.CommandQueue(ctx)
  time_ctx_queue_creation = time.time()

  mf = cl.mem_flags

  print('prepare device memory for input / output')

  ro_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ro)
  u_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
  B_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B)
  f_0_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f_0)
  f_1_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f_1)
  f_2_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f_2)

  time_devicedata_loaded = time.time()

  print('compile kernel code')
  prg = cl.Program(ctx, data).build(options=['-I', './c_sources'])
  time_kernel_compilation = time.time()

  knl = prg.solve
  knl_2 = prg.update_func

  print('prepare data ... ')
  initial()

  cl.enqueue_copy(queue, ro_g, ro)
  cl.enqueue_copy(queue, u_g, u)
  cl.enqueue_copy(queue, B_g, B)

  evt = knl_2(queue, (np.product(SHAPE),), None, u_g, B_g, ro_g, f_0_g, f_1_g, f_2_g)
  evt.wait()

  cl.enqueue_copy(queue, f_0, f_0_g)
  cl.enqueue_copy(queue, f_1, f_1_g)
  cl.enqueue_copy(queue, f_2, f_2_g)

  write_values(0)

  time_data_init = time.time()

  print('execute kernel programs\n')
  elapsed = 0.0
  write_values_time = 0.0
  read_values_time = 0.0

  for i in range(0, STEPS):
    if i % RW_DELETIMER == 0:
      read_start = time.time()
      read_values(i)
      read_values_time = time.time() - read_start

    cl.enqueue_copy(queue, ro_g, ro)
    cl.enqueue_copy(queue, u_g, u)
    cl.enqueue_copy(queue, B_g, B)
    cl.enqueue_copy(queue, f_0_g, f_0)
    cl.enqueue_copy(queue, f_1_g, f_1)
    cl.enqueue_copy(queue, f_2_g, f_2)

    evt = knl(queue, (np.product(SHAPE),), None, u_g, B_g, ro_g, f_0_g, f_1_g, f_2_g)
    # print('wait for kernel 1 executions')
    evt.wait()

    cl.enqueue_copy(queue, ro, ro_g)
    cl.enqueue_copy(queue, u, u_g)
    cl.enqueue_copy(queue, B, B_g)

    # elapsed += (1e-9 * (evt.profile.end - evt.profile.start))

    evt_2 = knl_2(queue, (np.product(SHAPE),), None, u_g, B_g, ro_g, f_0_g, f_1_g, f_2_g)
    # print('wait for kernel 2 executions')
    evt.wait()

    cl.enqueue_copy(queue, f_0, f_0_g)
    cl.enqueue_copy(queue, f_1, f_1_g)
    cl.enqueue_copy(queue, f_2, f_2_g)

    # elapsed += (1e-9 * (evt_2.profile.end - evt_2.profile.start))

    if (i+1) % RW_DELETIMER == 0:
      write_start = time.time()
      write_values(i+1)
      write_values_time += time.time() - write_start

  elapsed /= STEPS
  write_values_time /= int(STEPS / RW_DELETIMER)
  read_values_time /= int(STEPS / RW_DELETIMER)

  print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
  print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
  print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
  print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
  print('Initial data prepare took    : {}'.format(time_data_init - time_kernel_compilation))
  print('OpenCL elapsed time          : {}'.format(elapsed))
  print('OpenCL total time            : {}'.format(elapsed*STEPS))
  print('Write values elapsed time    : {}'.format(write_values_time))
  print('Write values total time      : {}'.format(write_values_time*int(STEPS / RW_DELETIMER)))
  print('Read values elapsed time     : {}'.format(read_values_time))
  print('Read values total time       : {}'.format(read_values_time*int(STEPS / RW_DELETIMER)))

  print('Total time                   : {}'.format(time.time() - start_time))

def compute_kinetic_energy(start_step, end_step):
    E = np.zeros((end_step-start_step)//RW_DELETIMER)
    it = 0
    x = list(range(start_step//RW_DELETIMER,end_step//RW_DELETIMER))
    for i in x:
        read_values(i*RW_DELETIMER)
        E[i] = 0.5 * np.sum(ro.reshape(scalar_shape)[0, :] * u.reshape(vec_shape)[0, :]**2)
    fig = plt.figure()
    ax = plt.axes()
    X = [i*DELTA_TAU for i in x]
    plt.plot(X, E)
    plt.savefig(Path(f"./{work_dir}/graphs/kin_energy.jpg"))

def compute_magnetic_energy(start_step, end_step):
    E = np.zeros((end_step-start_step)//RW_DELETIMER)
    it = 0
    x = list(range(start_step//RW_DELETIMER,end_step//RW_DELETIMER))
    for i in x:
        read_values(i*RW_DELETIMER)
        E[i] = 0.5 * np.sum([j**2 for j in B.reshape(vec_shape)[0, :]])
    fig = plt.figure()
    ax = plt.axes()
    X = [i*DELTA_TAU for i in x]
    plt.plot(X, E)
    plt.savefig(Path(f"./{work_dir}/graphs/mag_energy.jpg"))

def sum_ro(start_step, end_step):
    E = np.zeros((end_step-start_step)//RW_DELETIMER)
    it = 0
    x = list(range(start_step//RW_DELETIMER,end_step//RW_DELETIMER))
    for i in x:
        read_values(i*RW_DELETIMER)
        E[i] = np.sum([j for j in ro.reshape(scalar_shape)[0, :]])
    fig = plt.figure()
    ax = plt.axes()
    X = [i*DELTA_TAU for i in x]
    plt.plot(X, E)
    plt.savefig(Path(f"./{work_dir}/graphs/ro_sum.jpg"))
  
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

# main()

# compute_kinetic_energy(0, STEPS)

# compute_magnetic_energy(0, STEPS)

# sum_ro(0, STEPS)

# sum_quad_u(0, STEPS)

def test():

  import os
  os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
  os.environ['PYOPENCL_NO_CACHE'] = '1'

  DELTA_H = (2*3.14)/256

  foo_g = np.zeros(SHAPE).astype(np.float32)
  div = np.zeros(SHAPE).astype(np.float32)
  res = np.zeros(SHAPE).astype(np.float32)

  print('load program from cl source file')
  with open(Path('./c_sources/fd_math.cl'), 'r') as file:
    data = file.read()
    
  print('create context ...')
  platforms = cl.get_platforms()
  ctx = cl.Context(
    dev_type=cl.device_type.ALL,
    properties=[(cl.context_properties.PLATFORM, platforms[0])])

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

test()
