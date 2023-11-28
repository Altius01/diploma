import taichi as ti

from src.common.types import *
from src.common.matrix_ops import *

# @ti.kernel
# def knl_norm_field_sc(a: ti.template(), b:ti.template(), out: ti.template()):
#     for idx in ti.grouped(out):
#         out[idx] = norm_dot_sc(a[idx], b[idx])

# @ti.kernel
# def knl_norm_field_vec(a: ti.template(), b:ti.template(), out: ti.template()):
#     for idx in ti.grouped(out):
#         out[idx] = a[idx].dot(b[idx])

@ti.kernel
def knl_norm_field_mat(a: ti.template(), b:ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = norm_dot_mat(a[idx], b[idx])

@ti.kernel
def knl_norm_sqr_field_mat(a: ti.template(), b:ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = norm_sqr_dot_mat(a[idx], b[idx])

# def norm_field(a, b):
#     assert(a.shape == b.shape)
#     out = ti.field(dtype=a.dtype, shape=a.shape)
#     out.fill(0)

#     knl_norm_field(a, b, out)
#     return out

@ti.kernel
def knl_tr_field(a: ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = a.trace()

# def tr_field(a, b):
#     assert(a.shape == b.shape)
#     out = ti.field(dtype=a.dtype, shape=a.shape)
#     out.fill(0)

#     knl_tr_sqr_field(a, b, out)
#     return out

@ti.kernel
def knl_tr_sqr_field(a: ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = trace_sqr(a[idx])

@ti.kernel
def knl_tr_field(a: ti.template(), out: ti.template()):
    for idx in ti.grouped(out):
        out[idx] = trace(a[idx])

# def tr_sqr_field(a):
#     out = ti.field(dtype=a.dtype, shape=a.shape)
#     out.fill(0)

#     knl_tr_sqr_field(a, out)
#     return out

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

class Sum:
    s_sc = ti.field(double, shape=())
    s_vec = ti.Vector.field(n=3, dtype=double, shape=())
    s_mat = ti.Matrix.field(n=3, m=3, dtype=double, shape=())

    def sum_sc_field(field):
        Sum.s_sc[None] = 0
        knl_sum_field(field, Sum.s_sc)
        return  Sum.s_sc[None]

    def sum_vec_field(field):
        Sum.s_vec[None] = vec3(0)
        knl_sum_field(field, Sum.s_vec)
        return  Sum.s_vec[None]

    def sum_mat_field(field):
        Sum.s_mat[None] = mat3x3(0)
        knl_sum_field(field, Sum.s_mat)
        return  Sum.s_mat[None]
