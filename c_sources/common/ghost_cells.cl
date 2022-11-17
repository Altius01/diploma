#ifndef GHOST_CELLS
#define GHOST_CELLS

// __kernel void ghost_nodes_periodic(
//     global double *rho, global double *p, 
//     global double *u, global double *B
// ) {
//     int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

//     int4 index_x = (int4) {0, x, y, z};
//     int4 index_y = (int4) {1, x, y, z};
//     int4 index_z = (int4) {2, x, y, z};

//     if(is_not_ghost(index_x))
//         return;

//     int new_x = x;
//     int new_y = y;
//     int new_z = z;

//     if (x >= Nx-GHOST_CELLS)
//         new_x -= (Nx - 2*GHOST_CELLS);
//     else if (x < GHOST_CELLS)
//         new_x += (Nx - 2*GHOST_CELLS);

//     if (y >= Ny-GHOST_CELLS)
//         new_y -= (Ny - 2*GHOST_CELLS);
//     else if (y < GHOST_CELLS)
//         new_y += (Ny - 2*GHOST_CELLS);

//     if (z >= Nz-GHOST_CELLS)
//         new_z -= (Nz - 2*GHOST_CELLS);
//     else if (z < GHOST_CELLS)
//         new_z += (Nz - 2*GHOST_CELLS);


//     printf("Alarm!\n");

//     int4 new_index_x = (int4) {0, new_x, new_y, new_z};
//     int4 new_index_y = (int4) {1, new_x, new_y, new_z};
//     int4 new_index_z = (int4) {2, new_x, new_y, new_z};

//     // p[vec_buffer_idx(index_x)] = p[vec_buffer_idx(new_index_x)];
//     // rho[vec_buffer_idx(index_x)] = rho[vec_buffer_idx(new_index_x)];

//     // u[vec_buffer_idx(index_x)] = u[vec_buffer_idx(new_index_x)];
//     // u[vec_buffer_idx(index_y)] = u[vec_buffer_idx(new_index_y)];
//     // u[vec_buffer_idx(index_z)] = u[vec_buffer_idx(new_index_z)]; 

//     // B[vec_buffer_idx(index_x)] = B[vec_buffer_idx(new_index_x)];
//     // B[vec_buffer_idx(index_y)] = B[vec_buffer_idx(new_index_y)];
//     // B[vec_buffer_idx(index_z)] = B[vec_buffer_idx(new_index_z)];

//     p[vec_buffer_idx(index_x)] = -10;
//     rho[vec_buffer_idx(index_x)] = -10;

//     u[vec_buffer_idx(index_x)] = -10;
//     u[vec_buffer_idx(index_y)] = -10;
//     u[vec_buffer_idx(index_z)] = -10; 

//     B[vec_buffer_idx(index_x)] = -10;
//     B[vec_buffer_idx(index_y)] = -10;
//     B[vec_buffer_idx(index_z)] = -10;
// }

// __kernel void ghost_nodes_walls(
//     global double *rho, global double *p, 
//     global double *u, global double *B
// ) {
//     int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

//     int4 index_x = (int4) {0, x, y, z};
//     int4 index_y = (int4) {1, x, y, z};
//     int4 index_z = (int4) {2, x, y, z};

//     if(is_not_ghost(index_x))
//         return;

//     int new_x = x;
//     int new_y = y;
//     int new_z = z;

//     if (x >= Nx-GHOST_CELLS)
//         new_x = ((Nx - GHOST_CELLS) + (Nx - GHOST_CELLS - 1)) - x;
//     else if (x < GHOST_CELLS)
//         new_x = (2*GHOST_CELLS - 1  - x);

//     if (y >= Ny-GHOST_CELLS)
//         new_y = ((Ny - GHOST_CELLS) + (Ny - GHOST_CELLS - 1)) - y;
//     else if (y < GHOST_CELLS)
//         new_y = (2*GHOST_CELLS - 1  - y);

//     if (z >= Nz-GHOST_CELLS)
//         new_z = ((Nz - GHOST_CELLS) + (Nz - GHOST_CELLS - 1)) - z;
//     else if (z < GHOST_CELLS)
//         new_z = (2*GHOST_CELLS - 1  - z);

//     int4 new_index_x = (int4) {0, new_x, new_y, new_z};
//     int4 new_index_y = (int4) {1, new_x, new_y, new_z};
//     int4 new_index_z = (int4) {2, new_x, new_y, new_z};

//     p[vec_buffer_idx(index_x)] = p[vec_buffer_idx(new_index_x)];
//     rho[vec_buffer_idx(index_x)] = rho[vec_buffer_idx(new_index_x)];

//     u[vec_buffer_idx(index_x)] = -1.0*u[vec_buffer_idx(new_index_x)];
//     u[vec_buffer_idx(index_y)] = -1.0*u[vec_buffer_idx(new_index_y)];
//     u[vec_buffer_idx(index_z)] = -1.0*u[vec_buffer_idx(new_index_z)]; 

//     B[vec_buffer_idx(index_x)] = 0;
//     B[vec_buffer_idx(index_y)] = 0;
//     B[vec_buffer_idx(index_z)] = 0;
// }

// __kernel void test_boundaries(global double *arr) {
//     int x = get_global_id(0);
//     int N = get_global_size(0) - 4;
//     int new_x = x;

//     if (x < 2) {
//         new_x += N - 3;
//     } else if (x > N + 2) {
//         new_x -= N - 3;
//     }
    
//     arr[x] = arr[new_x];
// }

#endif