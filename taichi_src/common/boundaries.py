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
    
    for i in ti.static(range(3)):
        new_idx[i] = _get_ghost_new_idx(ghost, shape[i], idx[i])

    return new_idx