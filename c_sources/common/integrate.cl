#ifndef INTEGRATE
#define INTEGRATE

#include "mhd_consts.cl"

__kernel void kin_energy(
    global double *rho, global double *u, 
    global double *e
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int4 idx_e = (int4) {0, x, y, z};

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int4 index[3] = {index_x, index_y, index_z};

    e[t_vec_buffer_idx(idx_e)] = 0.5 * dV * rho[vec_buffer_idx(index_x)] * (
        u[vec_buffer_idx(index_x)]*u[vec_buffer_idx(index_x)] +
        u[vec_buffer_idx(index_y)]*u[vec_buffer_idx(index_y)] +
        u[vec_buffer_idx(index_z)]*u[vec_buffer_idx(index_z)]
    );
}

__kernel void mag_energy(
    global double *B, 
    global double *e
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int4 idx_e = (int4) {0, x, y, z};

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int4 index[3] = {index_x, index_y, index_z};

    e[t_vec_buffer_idx(idx_e)] = 0.5 * dV * (
        B[vec_buffer_idx(index_x)]*B[vec_buffer_idx(index_x)] +
        B[vec_buffer_idx(index_y)]*B[vec_buffer_idx(index_y)] +
        B[vec_buffer_idx(index_z)]*B[vec_buffer_idx(index_z)]
    );
}

#endif