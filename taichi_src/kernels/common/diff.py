import taichi as ti

# @ti.func
# def tensor_dot_vec3(a: ti.template(), b:ti.template()) -> mat3x3:
#     result = mat3x3(0)

#     for i, j in ti.ndrange(3, 3):
#         result[i, j] = a[i]*b[j]

#     return result

# type hints
double = ti.types.f64

vec3 = ti.types.vector(n=3, dtype=double)
vec5 = ti.types.vector(n=5, dtype=double)
mat3x3 = ti.types.matrix(n=3, m=3, dtype=double)

vec3i = ti.types.vector(n=3, dtype=int)
mat5x2i = ti.types.matrix(5, 2, int)
mat5x3i = ti.types.matrix(5, 3, int)
# type hints

kron = mat3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

@ti.func
def tensor_dot_vec3(a: ti.template(), b:ti.template()) -> mat3x3:
    return a.outer_product(b)

@ti.func
def get_diff_stencil(axe_1, axe_2, idx, order):
    result = mat5x3i([idx, idx, idx, idx, idx])
    stencil = mat5x2i(0)

    if order == 1:
        stencil = mat5x2i([(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)])
    elif order == 2:
        if axe_1 == axe_2:
            stencil = mat5x2i([(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)])
        else:
            stencil = mat5x2i([(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)])

    for i in ti.static(range(len(result))):
        _idx = idx
        if axe_1 == 0:
            _idx[0] += stencil[i, 0]
        elif axe_1 == 1:
            _idx[1] += stencil[i, 0]
        elif axe_1 == 2:
            _idx[2] += stencil[i, 0]

        if axe_2 == 0:
            _idx[0] += stencil[i, 1]
        elif axe_2 == 1:
            _idx[1] += stencil[i, 1]
        elif axe_2 == 2:
            _idx[2] += stencil[i, 1]

        for j in ti.static(range(3)):
            result[i, j] = _idx[j]
    
    return result

@ti.func
def get_diff_coefs(axe_1, axe_2, order):
    coefs = vec5(0)

    if order == 1:
        coefs = vec5([1.0/12, -8.0/12, 0.0, 8.0/12, -1.0/12])
    elif order == 2:
        if axe_1 == axe_2:
            coefs = vec5([-1.0/12, 16.0/12, -30.0/12, 16.0/12, -1.0/12])
        else:
            coefs = vec5([-1.0/4, -1.0/4, 0, -1.0/4, 1.0/4])

    return coefs

@ti.func
def get_weighted_sum(foo: ti.template(), axes: ti.template(), stencil: ti.template(), coefs: ti.template()):
    result = double(0.0)
    for i in ti.static(range(len(stencil))):
        idx = vec3i(0)
        for j in ti.static(range(3)):
            idx[j] = stencil[i, j]

        if ti.static(len(axes) == 1):
            result += foo(idx)[axes[0]] * coefs[i]
        elif ti.static(len(axes) == 2):
            mat_ = foo(idx)
            result += mat_[axes[0], axes[1]] * coefs[i]
        else:
            result += foo(idx) * coefs[i]
    return result

@ti.func
def dx(foo: ti.template(), axes, diff_axe: ti.i32, h:double, idx):
    coefs = get_diff_coefs(diff_axe, diff_axe, 1)
    stencil = get_diff_stencil(diff_axe, diff_axe, idx, 1)

    return get_weighted_sum(foo, axes, stencil, coefs) / h

@ti.func
def ddx(foo: ti.template(), axes, diff_axe_1: ti.i32, diff_axe_2: ti.i32, h1: double, h2: double, idx):
    coefs = get_diff_coefs(diff_axe_1, diff_axe_2, 2)
    stencil = get_diff_stencil(diff_axe_1, diff_axe_2, idx, 2)

    return get_weighted_sum(foo, axes, stencil, coefs) / h1 * h2

@ti.func
def grad(foo: ti.template(), h, idx):
    result = vec3(0)

    for i in ti.static(range(3)):
        result[i] = dx(foo, [], i, h[i], idx)
    return result

@ti.func
def div_vec3(foo: ti.template(), h, idx):
    result = double(0.0)

    for i in ti.static(range(3)):
        result += dx(foo, [i, ], i, h[i], idx)
    return result

@ti.func
def div_mat3x3(foo: ti.template(), axe, h, idx):
    result = vec3(0)

    for i, j in ti.ndrange(3, 3):
        if axe == 0:
            result[i] += dx(foo, [j, i], j, h[j], idx)
        elif axe == 1:
            result[i] += dx(foo, [i, j], j, h[j], idx)
    return result