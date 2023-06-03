import taichi as ti

@ti.func
def _get_ghost_new_idx(ghost, size, idx):
    new_idx = idx
    if idx < ghost:
        new_idx += (size - 2*ghost)
    elif idx >= size - ghost:
        new_idx -= (size - 2*ghost)
    return new_idx

@ti.func
def get_ghost_new_idx(ghost, shape, idx):
    new_idx = idx
    
    new_idx[0] = _get_ghost_new_idx(ghost[0], shape[0], idx[0])
    new_idx[1] = _get_ghost_new_idx(ghost[1], shape[1], idx[1])
    new_idx[2] = _get_ghost_new_idx(ghost[2], shape[2], idx[2])

    return new_idx