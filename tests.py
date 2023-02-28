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

GHOSTS = config.GHOSTS

T_SHAPE = config.T_SHAPE
SHAPE = config.SHAPE
FLUX_SHAPE = (T_SHAPE[0]+1, T_SHAPE[1]+1, T_SHAPE[2]+1,)

L = 2*np.pi
START_STEP = config.START_STEP
steps = config.STEPS
T =  steps * 0.5*(1/T_SHAPE[0])**2
RW_DELETIMER = config.RW_DELETIMER

scalar_shape = SHAPE
vec_shape = (3,) + SHAPE

# a = np.zeros(vec_shape).astype(np.float64)

def evaluate_ghosts(_knl, queue, arr):
  evt = _knl(queue, (2*GHOSTS, SHAPE[1], SHAPE[2]), None, np.int32(0), arr)
  evt.wait()

  evt = _knl(queue, (SHAPE[0], 2*GHOSTS, SHAPE[2]), None, np.int32(1), arr)
  evt.wait()

  evt = _knl(queue, (SHAPE[0], SHAPE[1], 2*GHOSTS), None, np.int32(2), arr)
  evt.wait()


def initials_dx_mul(a, b, c, ans_abc):
  for x in range(T_SHAPE[0]):
    for y in range(T_SHAPE[1]):
      for z in range(T_SHAPE[2]):
        _x = L*(1/T_SHAPE[0]) * (x - 0.5*T_SHAPE[0])

        a[x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_x)
        b[x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.exp(-_x**2)
        c[x+GHOSTS][y+GHOSTS][z+GHOSTS] = _x**2

        ans_abc[x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.exp(-_x**2) * _x * (_x*np.cos(_x) - 2*(_x**2 - 1)*np.sin(_x))


def initials_div(vec, ans):
  for x in range(T_SHAPE[0]):
    for y in range(T_SHAPE[1]):
      for z in range(T_SHAPE[2]):
        _x = L*(1/T_SHAPE[0]) * (x - 0.5*T_SHAPE[0])
        _y = L*(1/T_SHAPE[1]) * (y - 0.5*T_SHAPE[1])
        _z = L*(1/T_SHAPE[2]) * (z - 0.5*T_SHAPE[2])

        vec[0][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_x)*np.cos(_y)
        vec[1][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_y)*np.cos(_z)
        vec[2][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_z)*np.cos(_x)

        ans[x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.cos(_x)*np.cos(_y) + np.cos(_y)*np.cos(_z) + np.cos(_z)*np.cos(_x)

  
def initials_rot(vec, ans):
  for x in range(T_SHAPE[0]):
    for y in range(T_SHAPE[1]):
      for z in range(T_SHAPE[2]):
        _x = L*(1/T_SHAPE[0]) * (x - 0.5*T_SHAPE[0])
        _y = L*(1/T_SHAPE[1]) * (y - 0.5*T_SHAPE[1])
        _z = L*(1/T_SHAPE[2]) * (z - 0.5*T_SHAPE[2])

        vec[0][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_x)*np.cos(_y)
        vec[1][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_y)*np.cos(_z)
        vec[2][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_z)*np.cos(_x)

        ans[0][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_y)*np.sin(_z)
        ans[1][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_z)*np.sin(_x)
        ans[2][x+GHOSTS][y+GHOSTS][z+GHOSTS] = np.sin(_x)*np.sin(_y)


def test_dx_mul(knl_dx_mul, knl_ghosts, ctx, mf, queue):
  a = np.zeros(scalar_shape).astype(np.float64)
  b = np.zeros(scalar_shape).astype(np.float64)
  c = np.zeros(scalar_shape).astype(np.float64)

  test_ans = np.zeros(scalar_shape).astype(np.float64)
  ans_abc = np.zeros(scalar_shape).astype(np.float64)

  a_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
  b_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=b)
  c_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c)
  ans_abc_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ans_abc)

  initials_dx_mul(a, b, c, ans_abc)

  cl.enqueue_copy(queue, a_gpu, a)
  cl.enqueue_copy(queue, b_gpu, b)
  cl.enqueue_copy(queue, c_gpu, c)
  cl.enqueue_copy(queue, ans_abc_gpu, ans_abc)

  evaluate_ghosts(knl_ghosts, queue, a_gpu)
  evaluate_ghosts(knl_ghosts, queue, b_gpu)
  evaluate_ghosts(knl_ghosts, queue, c_gpu)
  evaluate_ghosts(knl_ghosts, queue, ans_abc_gpu)

  cl.enqueue_copy(queue, a, a_gpu)
  cl.enqueue_copy(queue, b, b_gpu)
  cl.enqueue_copy(queue, c, c_gpu)
  cl.enqueue_copy(queue, ans_abc, ans_abc_gpu)

  evt = knl_dx_mul(queue, scalar_shape, None, a_gpu, b_gpu, c_gpu, ans_abc_gpu)
  evt.wait()

  evaluate_ghosts(knl_ghosts, queue, ans_abc_gpu)

  cl.enqueue_copy(queue, test_ans, ans_abc_gpu)
  
  # X=  np.linspace(0, L, SHAPE[0])
  X = list(range(0, SHAPE[0]))
  plt.plot(X, ans_abc[:, 0, 0], '-ro')
  plt.plot(X, test_ans[:, 0, 0], '-go')
  plt.show()
  print(np.allclose(ans_abc, test_ans, atol=(L/T_SHAPE[0])**(4)))
  assert(np.allclose(ans_abc, test_ans, atol=(L/T_SHAPE[0])**(4)))


def test_div(knl_div, knl_sc_ghosts, knl_vec_ghosts, ctx, mf, queue):
  vec = np.zeros(vec_shape).astype(np.float64)
  ans = np.zeros(scalar_shape).astype(np.float64)
  test_ans = np.zeros(scalar_shape).astype(np.float64)

  vec_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vec)
  ans_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ans)

  initials_div(vec, ans)

  cl.enqueue_copy(queue, vec_gpu, vec)
  cl.enqueue_copy(queue, ans_gpu, ans)

  evaluate_ghosts(knl_vec_ghosts, queue, vec_gpu)
  evaluate_ghosts(knl_vec_ghosts, queue, ans_gpu)

  cl.enqueue_copy(queue, vec, vec_gpu)
  cl.enqueue_copy(queue, ans, ans_gpu)

  evt = knl_div(queue, scalar_shape, None, vec_gpu, ans_gpu)
  evt.wait()

  evaluate_ghosts(knl_vec_ghosts, queue, ans_gpu)

  cl.enqueue_copy(queue, test_ans, ans_gpu)
  
  X = list(range(0, SHAPE[0]))
  plt.plot(X, vec[0, :, 3, 3], '-bo')
  plt.plot(X, ans[:, 3, 3], '-ro')
  plt.plot(X, test_ans[:, 3, 3], '-go')
  plt.show()
  print(np.allclose(ans, test_ans, atol=(L/T_SHAPE[0])**(4)))
  assert(np.allclose(ans, test_ans, atol=(L/T_SHAPE[0])**(4)))


def test_rot(knl_rot, knl_sc_ghosts, knl_vec_ghosts, ctx, mf, queue):
  vec = np.zeros(vec_shape).astype(np.float64)
  ans = np.zeros(vec_shape).astype(np.float64)
  test_ans = np.zeros(vec_shape).astype(np.float64)

  vec_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vec)
  ans_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=ans)

  initials_rot(vec, ans)

  cl.enqueue_copy(queue, vec_gpu, vec)
  cl.enqueue_copy(queue, ans_gpu, ans)

  evaluate_ghosts(knl_vec_ghosts, queue, vec_gpu)
  evaluate_ghosts(knl_vec_ghosts, queue, ans_gpu)

  cl.enqueue_copy(queue, vec, vec_gpu)
  cl.enqueue_copy(queue, ans, ans_gpu)

  evt = knl_rot(queue, scalar_shape, None, vec_gpu, ans_gpu)
  evt.wait()

  evaluate_ghosts(knl_vec_ghosts, queue, ans_gpu)

  cl.enqueue_copy(queue, test_ans, ans_gpu)
  
  X = list(range(0, SHAPE[0]))
  # plt.plot(X, vec[0, :, 3, 3], '-bo')
  plt.plot(X, ans[0, :, 16, 16], '-ro')
  plt.plot(X, test_ans[0, :, 16, 16], '-go')
  plt.show()
  print(np.allclose(ans, test_ans, atol=(L/T_SHAPE[0])**(4)))
  assert(np.allclose(ans, test_ans, atol=(L/T_SHAPE[0])**(4)))
  


def main():
  global u, B_arr, rho, e_kin, e_mag

  data_service = DataService("TESTS_" + str(date.today()), scalar_shape, vec_shape)

  with open(Path('./c_sources/tests.cl'), 'r') as file:
    data = file.read()
    
  ctx = cl.create_some_context()

  queue = cl.CommandQueue(ctx)

  mf = cl.mem_flags

  prg = cl.Program(ctx, data).build(options=['-I', './c_sources'])

  knl_div = prg.test_div
  knl_rot = prg.test_rot
  knl_dx_mul = prg.test_dx_mul
  knl_vec_ghosts = prg.vec_ghost_nodes_periodic
  knl_sc_ghosts = prg.sc_ghost_nodes_periodic

  test_rot(knl_rot, knl_sc_ghosts, knl_vec_ghosts, ctx, mf, queue)
  # test_div(knl_div, knl_sc_ghosts, knl_vec_ghosts, ctx, mf, queue)
  # test_dx_mul(knl_dx_mul, knl_sc_ghosts, ctx, mf, queue)


if __name__ == "__main__":
    main()
