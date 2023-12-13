from abc import ABC, abstractclassmethod
import taichi as ti
from src.common.matrix_ops import get_basis

from src.flux.limiters import minmod


class Reconstructor(ABC):
    def __init__(self, axis):
        self.axis = axis

    @abstractclassmethod
    def get_right(self, q: ti.template(), idx):
        raise NotImplementedError

    @abstractclassmethod
    def get_left(self, q: ti.template(), idx):
        raise NotImplementedError


from src.common.types import *


class FirstOrder(Reconstructor):
    @ti.func
    def get_right(self, q: ti.template(), idx):
        return vec3(0) + q(idx)

    @ti.func
    def get_left(self, q: ti.template(), idx):
        return vec3(0) + q(idx + get_basis(self.axis))


class SecondOrder(Reconstructor):
    @ti.func
    def get_right(self, q: ti.template(), idx):
        return vec3(0) + tvd_slope_limiter_2order(
            q, idx + get_basis(self.axis), self.axis
        )

    @ti.func
    def get_left(self, q: ti.template(), idx):
        return vec3(0) + tvd_slope_limiter_2order(q, idx, self.axis)


class Weno5(Reconstructor):
    @ti.func
    def get_right(self, q: ti.template(), idx):
        return vec3(0) + get_weno(q, idx + get_basis(self.axis), self.axis)

    @ti.func
    def get_left(self, q: ti.template(), idx):
        return vec3(0) + get_weno(q, idx, self.axis)


@ti.func
def get_weno(foo: ti.template(), idx, i):
    fm2 = foo(idx - 2 * get_basis(i))
    fm1 = foo(idx - get_basis(i))
    f = foo(idx)
    fp1 = foo(idx + get_basis(i))
    fp2 = foo(idx + 2 * get_basis(i))

    g1 = 0.1
    g2 = 0.6
    g3 = 0.3

    b1 = (13.0 / 12.0) * (fm2 - 2 * fm1 + f) ** 2 + 0.25 * (fm2 - 4 * fm1 + 3 * f) ** 2
    b2 = (13.0 / 12.0) * (fm1 - 2 * f + fp1) ** 2 + 0.25 * (fm1 - fp1) ** 2
    b3 = (13.0 / 12.0) * (f - 2 * fp1 + fp2) ** 2 + 0.25 * (3 * f - 4 * fp1 + fp2) ** 2

    a1 = g1 / (b1 + 1e-6) ** 2
    a2 = g2 / (b2 + 1e-6) ** 2
    a3 = g3 / (b3 + 1e-6) ** 2

    a_sum = a1 + a2 + a3

    w1 = a1 / a_sum
    w2 = a1 / a_sum
    w3 = a1 / a_sum

    return (
        w1 * ((1.0 / 3.0) * fm2 - (7.0 / 6.0) * fm1 + (11.0 / 6.0) * f)
        + w2 * (-(1.0 / 6.0) * fm1 + (5.0 / 6.0) * f + (1.0 / 3.0) * fp1)
        + w3 * ((1.0 / 3.0) * f + (5.0 / 6.0) * fp1 - (1.0 / 6.0) * fp2)
    )


@ti.func
def tvd_slope_limiter_2order(
    q: ti.template(),
    idx: int,
    axes: int = 0,
    k: float = -1.0,
    limiter: ti.template() = minmod,
):
    idx_left = idx
    idx_right = idx + 2 * get_basis(axes)
    idx_new = idx + get_basis(axes)

    D_m = q(idx_new) - q(idx_left)
    D_p = q(idx_right) - q(idx_new)

    return q(idx) + 0.25 * (
        (1 - k) * limiter(D_p / D_m) * D_m + (1 + k) * limiter(D_m / D_p) * D_p
    )
