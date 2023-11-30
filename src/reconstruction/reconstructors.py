from abc import ABC, abstractclassmethod
import taichi as ti
from src.common.matrix_ops import get_basis

from src.flux.limiters import minmod


class Reconstructor(ABC):
    def __init__(self, axis):
        self.axis = axis

    @ti.func
    @abstractclassmethod
    def get_right(self, q: ti.template(), idx):
        ...

    @ti.func
    @abstractclassmethod
    def get_left(self, q: ti.template(), idx):
        ...


class FirstOrder(Reconstructor):
    @ti.func
    @abstractclassmethod
    def get_right(self, q: ti.template(), idx):
        return q(idx)

    @ti.func
    @abstractclassmethod
    def get_left(self, q: ti.template(), idx):
        return q(idx)


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