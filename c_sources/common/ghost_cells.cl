#ifndef GHOSTS
#define GHOSTS

#include "mhd_consts.cl"

int get_ghost_idx(int idx, const int size, const int ghosts) {
    while (idx >= size-ghosts)
        idx -= (size - 2*ghosts);

    while (idx < ghosts)
        idx += (size - 2*ghosts);

    return idx;
}

__kernel void vec_ghost_nodes_periodic(
    int ax,
    global double *arr
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int sizes[3] = {Nx, Ny, Nz};
    int axes[3] = {x, y, z};
    axes[ax] += (axes[ax] >= GHOST_CELLS) ? sizes[ax] - 2*GHOST_CELLS : 0;

    x = axes[0];
    y = axes[1];
    z = axes[2];

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int new_x = get_ghost_idx(x, Nx, GHOST_CELLS);
    int new_y = get_ghost_idx(y, Ny, GHOST_CELLS);
    int new_z = get_ghost_idx(z, Nz, GHOST_CELLS);
    
    int4 new_index_x = (int4) {0, new_x, new_y, new_z};
    int4 new_index_y = (int4) {1, new_x, new_y, new_z};
    int4 new_index_z = (int4) {2, new_x, new_y, new_z};

    arr[vec_buffer_idx(index_x)] = arr[vec_buffer_idx(new_index_x)];
    arr[vec_buffer_idx(index_y)] = arr[vec_buffer_idx(new_index_y)];
    arr[vec_buffer_idx(index_z)] = arr[vec_buffer_idx(new_index_z)];
}

__kernel void sc_ghost_nodes_periodic(
    int ax,
    global double *arr
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int sizes[3] = {Nx, Ny, Nz};
    int axes[3] = {x, y, z};
    axes[ax] += (axes[ax] >= GHOST_CELLS) ? sizes[ax] - 2*GHOST_CELLS : 0;

    x = axes[0];
    y = axes[1];
    z = axes[2];

    int4 index_x = (int4) {0, x, y, z};

    int new_x = get_ghost_idx(x, Nx, GHOST_CELLS);
    int new_y = get_ghost_idx(y, Ny, GHOST_CELLS);
    int new_z = get_ghost_idx(z, Nz, GHOST_CELLS);
    
    int4 new_index_x = (int4) {0, new_x, new_y, new_z};

    arr[vec_buffer_idx(index_x)] = arr[vec_buffer_idx(new_index_x)];
}

__kernel void ghost_nodes_periodic(
    int ax,
    global double *rho, global double *p, 
    global double *u, global double *B
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int sizes[3] = {Nx, Ny, Nz};
    int axes[3] = {x, y, z};
    axes[ax] += (axes[ax] >= GHOST_CELLS) ? sizes[ax] - 2*GHOST_CELLS : 0;

    x = axes[0];
    y = axes[1];
    z = axes[2];

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int new_x = get_ghost_idx(x, Nx, GHOST_CELLS);
    int new_y = get_ghost_idx(y, Ny, GHOST_CELLS);
    int new_z = get_ghost_idx(z, Nz, GHOST_CELLS);
    
    int4 new_index_x = (int4) {0, new_x, new_y, new_z};
    int4 new_index_y = (int4) {1, new_x, new_y, new_z};
    int4 new_index_z = (int4) {2, new_x, new_y, new_z};

    p[vec_buffer_idx(index_x)] = p[vec_buffer_idx(new_index_x)];
    rho[vec_buffer_idx(index_x)] = rho[vec_buffer_idx(new_index_x)];

    u[vec_buffer_idx(index_x)] = u[vec_buffer_idx(new_index_x)];
    u[vec_buffer_idx(index_y)] = u[vec_buffer_idx(new_index_y)];
    u[vec_buffer_idx(index_z)] = u[vec_buffer_idx(new_index_z)]; 

    B[vec_buffer_idx(index_x)] = B[vec_buffer_idx(new_index_x)];
    B[vec_buffer_idx(index_y)] = B[vec_buffer_idx(new_index_y)];
    B[vec_buffer_idx(index_z)] = B[vec_buffer_idx(new_index_z)];
}

#endif