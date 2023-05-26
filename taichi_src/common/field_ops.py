import taichi as ti

from taichi_src.common.types import *
from taichi_src.common.matrix_ops import norm_dot, trace_sqr

@ti.kernel
def knl_norm_field(a: ti.template(), b:ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = norm_dot(a[idx], b[idx])

def norm_field(a, b):
    assert(a.shape == b.shape)
    out = ti.field(dtype=a.dtype, shape=a.shape)
    out.fill(0)

    knl_norm_field(a, b, out)
    return out

@ti.kernel
def knl_tr_field(a: ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = a.trace()

def tr_field(a, b):
    assert(a.shape == b.shape)
    out = ti.field(dtype=a.dtype, shape=a.shape)
    out.fill(0)

    knl_tr_sqr_field(a, b, out)
    return out

@ti.kernel
def knl_tr_sqr_field(a: ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = trace_sqr(a[idx])

def tr_sqr_field(a):
    out = ti.field(dtype=a.dtype, shape=a.shape)
    out.fill(0)

    knl_tr_sqr_field(a, out)
    return out

@ti.kernel
def knl_field_div(a: ti.template(), b:ti.template()):
    for idx in ti.grouped(a):
        a[idx] = a[idx] / b[idx]

def field_div(a, b):
    assert(a.shape == b.shape)
    knl_field_div(a, b)

@ti.kernel
def knl_sum_field(field: ti.template(), s: ti.template()):
    for idx in ti.grouped(field):
        s[None] += field[idx]

def sum_sc_field(field):
    s = ti.field(double, shape=())

    knl_sum_field(field, s)
    return s[None]

def sum_vec_field(field):
    s = ti.Vector.field(n=3, dtype=double, shape=())

    knl_sum_field(field, s)
    return s[None]

def sum_mat_field(field):
    s = ti.Matrix.field(n=3, m=3, dtype=double, shape=())

    knl_sum_field(field, s)
    return s[None]
