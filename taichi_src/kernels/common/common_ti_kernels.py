import taichi as ti

@ti.func
def ti_kron(i: ti.i32, j: ti.i32) -> ti.f64:
    result = 0
    if i == j:
        result = 1
    return result

@ti.func
def get_sc_idx(vec_idx):
    result = vec_idx[:3]

    for i in range(len(vec_idx)-3, len(vec_idx)):
        result[i] = vec_idx[i]
    return result

@ti.kernel
def spatial_mean(a: ti.types.ndarray(), result: ti.template(), dV: ti.f64):
    assert len(result.shape) == 0
    for i in ti.grouped(a):
        result[None] += a[i]*dV

@ti.kernel
def mat_dot(a: ti.types.ndarray(), 
            b: ti.types.ndarray(), 
            result: ti.types.ndarray()):
    assert a.shape == b.shape
    assert len(a.shape) == 5
    assert len(result.shape) == 3

    for idx in ti.grouped(result):
        x, y, z = idx
        for i, j in ti.ndrange(a.shape[0], a.shape[1]):
            result[idx] += a[i, j, x, y, z]*b[i, j, x, y, z]

@ti.kernel
def mat_trace_dot(a: ti.types.ndarray(), 
            b: ti.types.ndarray(), 
            result: ti.types.ndarray()):
    assert len(a.shape) == 5
    assert len(result.shape) == 3

    for idx in ti.grouped(result):
        x, y, z = idx
        for i in ti.ndrange(3):
            result[idx] += a[i, i, x, y, z]*b[i, i, x, y, z]

@ti.func
def f_mat_abs_quad(a: ti.types.ndarray(), result: ti.types.ndarray()):
    assert len(a.shape) == 5
    assert len(result.shape) == 3

    for idx in ti.grouped(result):
        x, y, z = idx
        for i, j in ti.ndrange(a.shape[0], a.shape[1]):
            result[idx] += a[i, j, x, y, z]*a[i, j, x, y, z]

@ti.func
def mat_abs_quad(a: ti.types.ndarray(), result: ti.types.ndarray()):
    f_mat_abs_quad(a, result)

@ti.kernel
def mat_abs(a: ti.types.ndarray(), result: ti.types.ndarray()):
    f_mat_abs_quad(a, result)

    for idx in ti.grouped(result):
        result[idx] = ti.sqrt(result[idx])