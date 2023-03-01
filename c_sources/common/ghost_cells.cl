#ifndef GHOSTS
#define GHOSTS

#include "mhd_consts.cl"

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

    int new_x = x;
    int new_y = y;
    int new_z = z;

    if (x >= Nx-GHOST_CELLS)
        new_x -= (Nx - 2*GHOST_CELLS);
    else if (x < GHOST_CELLS)
        new_x += (Nx - 2*GHOST_CELLS);

    if (y >= Ny-GHOST_CELLS)
        new_y -= (Ny - 2*GHOST_CELLS);
    else if (y < GHOST_CELLS)
        new_y += (Ny - 2*GHOST_CELLS);

    if (z >= Nz-GHOST_CELLS)
        new_z -= (Nz - 2*GHOST_CELLS);
    else if (z < GHOST_CELLS)
        new_z += (Nz - 2*GHOST_CELLS);
    
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

    int new_x = x;
    int new_y = y;
    int new_z = z;

    if (x >= Nx-GHOST_CELLS)
        new_x -= (Nx - 2*GHOST_CELLS);
    else if (x < GHOST_CELLS)
        new_x += (Nx - 2*GHOST_CELLS);

    if (y >= Ny-GHOST_CELLS)
        new_y -= (Ny - 2*GHOST_CELLS);
    else if (y < GHOST_CELLS)
        new_y += (Ny - 2*GHOST_CELLS);

    if (z >= Nz-GHOST_CELLS)
        new_z -= (Nz - 2*GHOST_CELLS);
    else if (z < GHOST_CELLS)
        new_z += (Nz - 2*GHOST_CELLS);
    
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
    axes[ax] += (axes[ax] > 2) ? sizes[ax] - 2*GHOST_CELLS : 0;

    x = axes[0];
    y = axes[1];
    z = axes[2];

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int new_x = x;
    int new_y = y;
    int new_z = z;

    if (x >= Nx-GHOST_CELLS)
        new_x -= (Nx - 2*GHOST_CELLS);
    else if (x < GHOST_CELLS)
        new_x += (Nx - 2*GHOST_CELLS);

    if (y >= Ny-GHOST_CELLS)
        new_y -= (Ny - 2*GHOST_CELLS);
    else if (y < GHOST_CELLS)
        new_y += (Ny - 2*GHOST_CELLS);

    if (z >= Nz-GHOST_CELLS)
        new_z -= (Nz - 2*GHOST_CELLS);
    else if (z < GHOST_CELLS)
        new_z += (Nz - 2*GHOST_CELLS);
    
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