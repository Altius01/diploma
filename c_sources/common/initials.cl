#ifndef INITIALS
#define INITIALS

#include "mhd_consts.cl"

__kernel void Orszag_Tang_3D_inital(
    __global double *_rho, __global double *_p, 
    __global double *_u, __global double *_B
) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    if (is_not_ghost(index_x)) {
        x -= GHOST_CELLS;
        y -= GHOST_CELLS;
        z -= GHOST_CELLS;

        _p[vec_buffer_idx(index_x)] = 5.0/(12.0*M_PI);
        _rho[vec_buffer_idx(index_x)] = 25.0/(36.0*M_PI);

        // (1 + eps_p*sin(2.0*M_PI*hz*z))
        _u[vec_buffer_idx(index_x)] = -1.0*(1 + eps_p*sin(hz*z))*u0*sin(hy*y);
        _u[vec_buffer_idx(index_y)]  = (1 + eps_p*sin(hz*z))*u0*sin(hx*x);
        _u[vec_buffer_idx(index_z)]  = eps_p*sin(hz*z);

        _B[vec_buffer_idx(index_x)]  = -B0*sin(hy*y);
        _B[vec_buffer_idx(index_y)]  = B0*sin(2.0*hx*x);
        _B[vec_buffer_idx(index_z)]  = 0;

        // _B[vec_buffer_idx(index_x)]  = sin(hx*x);
        // _B[vec_buffer_idx(index_y)]  = 0;
        // _B[vec_buffer_idx(index_z)]  = 0;

        // _u[vec_buffer_idx(index_x)] = 0;
        // _u[vec_buffer_idx(index_y)] = 0;
        // _u[vec_buffer_idx(index_z)] = 0;

        // _B[vec_buffer_idx(index_x)] = 0;
        // _B[vec_buffer_idx(index_y)] = 0;
        // _B[vec_buffer_idx(index_z)] = 0;

        // if (x*hx < 0.5) {
        //     _p[vec_buffer_idx(index_x)] = 1.0;
        //     _rho[vec_buffer_idx(index_x)] = 1.0;
        // }
        // else {
        //     _p[vec_buffer_idx(index_x)] = pow(0.125, gamma);
        //     _rho[vec_buffer_idx(index_x)] = 0.125;
        // }
    }
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