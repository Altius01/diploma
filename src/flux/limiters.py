import taichi as ti


@ti.func
def minmod(r):
    return ti.max(ti.min(r, 1), 0)
