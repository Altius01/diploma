#ifndef INITIALS
#define INITIALS

#include "mhd_consts.cl"

// bool is_not_ghost(int4 i) {
//     if ((i.s1 >= GHOST_CELLS && i.s1 < Nx-GHOST_CELLS) 
//         && (i.s2 >= GHOST_CELLS && i.s2 < Ny-GHOST_CELLS) 
//         && (i.s3 >= GHOST_CELLS && i.s3 < Nz-GHOST_CELLS))
//         return true;
//     else
//         return false;
// }

// bool is_flux(int4 i) {
//     if ((i.s1 >= GHOST_CELLS-1 && i.s1 < Nx-GHOST_CELLS) 
//         && (i.s2 >= GHOST_CELLS-1 && i.s2 < Ny-GHOST_CELLS) 
//         && (i.s3 >= GHOST_CELLS-1 && i.s3 < Nz-GHOST_CELLS))
//         return true;
//     else
//         return false;
// }

__kernel void Orszag_Tang_3D_inital(
    __global double *rho, __global double *p, 
    __global double *u, __global double *B
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    x -= GHOST_CELLS;
    y -= GHOST_CELLS;
    z -= GHOST_CELLS;

    // p[vec_buffer_idx(index_x)] = 5.0/(12.0*M_PI);
    // rho[vec_buffer_idx(index_x)] = 25.0/(36.0*M_PI);

    // u[vec_buffer_idx(index_x)] = -1.0*(1 + epsp*sin(hz*z))*u0*sin(hy*y);
    // u[vec_buffer_idx(index_y)]  = (1 + epsp*sin(hz*z))*u0*sin(hx*x);
    // u[vec_buffer_idx(index_z)]  = epsp*sin(hz*z);

    // B[vec_buffer_idx(index_x)]  = -B0*sin(hy*y);
    // B[vec_buffer_idx(index_y)]  = B0*sin(2.0*hx*x);
    // B[vec_buffer_idx(index_z)]  = 0;

    p[vec_buffer_idx(index_x)] = 1;
    rho[vec_buffer_idx(index_x)] = 1;

    u[vec_buffer_idx(index_x)] = -1.0*(1 + eps_p*sin(hz*z))*sin(hy*y);
    u[vec_buffer_idx(index_y)]  = (1 + eps_p*sin(hz*z))*sin(hx*x);
    u[vec_buffer_idx(index_z)]  = eps_p*sin(hz*z);

    B[vec_buffer_idx(index_x)]  = -sin(hy*y);
    B[vec_buffer_idx(index_y)]  = sin(2.0*hx*x);
    B[vec_buffer_idx(index_z)]  = 0;
}

__kernel void Tanh_3D_inital(
    __global double *rho, __global double *p, 
    __global double *u, __global double *B
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    rho[vec_buffer_idx(index_x)] = tanh(hx*x) + tanh(hy*y) + tanh(hz*z);
    p[vec_buffer_idx(index_x)] = pow(rho[vec_buffer_idx(index_x)], gamma);

    u[vec_buffer_idx(index_x)] = tanh(hx*x) + tanh(hy*y) + tanh(hz*z);
    u[vec_buffer_idx(index_y)] = tanh(hx*x) + tanh(hy*y) + tanh(hz*z);
    u[vec_buffer_idx(index_z)] = tanh(hx*x) + tanh(hy*y) + tanh(hz*z);

    B[vec_buffer_idx(index_x)] = tanh(hx*x) + tanh(hy*y) + tanh(hz*z);
    B[vec_buffer_idx(index_y)] = tanh(hx*x) + tanh(hy*y) + tanh(hz*z);
    B[vec_buffer_idx(index_z)] = tanh(hx*x) + tanh(hy*y) + tanh(hz*z);
}

__kernel void test_inital(
    __global double *arr
) {
    int x = get_global_id(0);
    double hx = 1.0/(get_global_size(0)-4);
    
    arr[x + 2] = cos(2.0*M_PI*hx*x);
}

#endif