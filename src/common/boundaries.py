from re import I
import taichi as ti

from src.common.pointers import get_elem_1d


@ti.func
def _get_ghost_new_idx(ghost, size, idx):
    new_idx = idx

    true_size = size - 2 * ghost
    if idx < ghost:
        new_idx += true_size

    if idx >= size - ghost:
        new_idx -= true_size

    return new_idx


@ti.func
def get_ghost_new_idx(ghost, shape, idx, dim=2):
    new_idx = idx

    for i in range(dim):
        new_idx[i] = _get_ghost_new_idx(
            get_elem_1d(ghost, i), get_elem_1d(shape, i), get_elem_1d(idx, i)
        )

    return new_idx


@ti.func
def _get_mirror_new_idx(ghost, size, idx):
    new_idx = idx
    if idx < ghost:
        new_idx = 2 * ghost - (idx + 1)
    elif idx >= size - ghost:
        new_idx = 2 * (size - ghost) - (idx + 1)
    return new_idx


@ti.func
def get_mirror_new_idx(ghost, shape, idx):
    new_idx = idx

    new_idx[0] = _get_mirror_new_idx(ghost[0], shape[0], idx[0])
    new_idx[1] = _get_mirror_new_idx(ghost[1], shape[1], idx[1])
    new_idx[2] = _get_mirror_new_idx(ghost[2], shape[2], idx[2])

    return new_idx


@ti.func
def _check_ghost(shape, ghost, idx):
    return (idx < ghost) or (idx >= shape - ghost)
