import taichi as ti

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
            l[2:] = l[2:] * filter_size + j[:]
            ti.atomic_add(out[i], src[l])
        out[i] /= filter_size**3
