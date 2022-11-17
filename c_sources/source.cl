#include "fd_math.cl"

MYFLOAT f_rho(__global MYFLOAT *rho, __global MYFLOAT *u, int3 idx) {
    MYFLOAT result = 0.0f;
    int3 axes = {0, 0, 0};
    for (int j = 0; j < 3; ++j) {
        axes.s0 = j; axes.s2 = j;
        result += prod2_arr_diff_i(u, rho, axes, idx);
    }
    return result;
}

MYFLOAT f_u(__global MYFLOAT *rho, __global MYFLOAT *u, __global MYFLOAT *B, FOO_VEC p, int i, int3 idx) {
    MYFLOAT result = 0.0f;
    int3 ub = {0, 0, 0};
    int4 ruu = {0, 0, 0, 0};

    ruu.s2 = i;
    ub.s1 = i;

    for (int j = 0; j < 3; ++j) {
        ruu.s0 = j; ruu.s3 = j; ub.s0 = j; ub.s2 = j;
        result += prod3_arr_diff_i(rho, u, u, ruu, idx) + 
            (Diff_i(p, j, 0, idx.x, idx.y, idx.z) + abs_prod_arr_diff(B, j, idx)/(2*MA*MAXFLOAT))*kron(i, j) - 
            sigma(u, s_second_arr, i, j, idx.x, idx.y, idx.z)/RE - prod2_arr_diff_i(B, B, ub, idx)/(MA*MA);
    }
    return result;
}

MYFLOAT f_B(__global MYFLOAT *u, __global MYFLOAT *B, int i, int3 idx) {
    MYFLOAT result = 0.0f;
    int3 ub = {0, 0, 0};
    ub.s1 = i;
    for (int j = 0; j < 3; ++j) {
        ub.s0 = j; ub.s2 = j;
        result += prod2_arr_diff_i(B, u, ub, idx) - prod2_arr_diff_i(u, B, ub, idx) - 
            Diff_arr_ii(B, j, i, idx.x, idx.y, idx.z);
    }
    return result;
}

__global MYFLOAT rk1[SIZE_X*SIZE_Y*SIZE_Z];
__global MYFLOAT rk2[SIZE_X*SIZE_Y*SIZE_Z];
__global MYFLOAT rk3[SIZE_X*SIZE_Y*SIZE_Z];

__global MYFLOAT uk1[3*SIZE_X*SIZE_Y*SIZE_Z];
__global MYFLOAT uk2[3*SIZE_X*SIZE_Y*SIZE_Z]; 
__global MYFLOAT uk3[3*SIZE_X*SIZE_Y*SIZE_Z]; 

__global MYFLOAT Bk1[3*SIZE_X*SIZE_Y*SIZE_Z];
__global MYFLOAT Bk2[3*SIZE_X*SIZE_Y*SIZE_Z]; 
__global MYFLOAT Bk3[3*SIZE_X*SIZE_Y*SIZE_Z]; 

__kernel void solve_system(
    __global MYFLOAT *u, __global MYFLOAT *B, __global MYFLOAT *rho,
    __global MYFLOAT *new_arr_u, __global MYFLOAT *new_arr_B, __global MYFLOAT *new_arr_rho
    ) {
        int x = get_global_id(0); int y = get_global_id(1); 
        int z = get_global_id(2);

        int3 idx = 0; idx.x = x; idx.y = y; idx.z = z;
        int b_idx = buffer_idx(x, y, z);

        int v_idx[3];
        for(int i = 0; i < 3; ++i)
            v_idx[i] = vec_buffer_idx(i, x, y, z);

        FOO_VEC p = ^(int i, int x, int y, int z) {
            MYFLOAT ret = rho[buffer_idx(x, y, z)];
            return ret;
        };

        MYFLOAT temp = f_rho(rho, u, idx);
        new_arr_rho[b_idx] = rho[b_idx] + (DELTA_TAU/6) * temp;
        rk1[b_idx] = rho[b_idx] + (DELTA_TAU/2) * temp;
        
        for(int i = 0; i < 3; ++i) {
            temp = f_u(rho, u, B, p, i, idx);
            new_arr_u[v_idx[i]] = u[v_idx[i]] + (DELTA_TAU/6) * temp;
            uk1[v_idx[i]] = u[v_idx[i]] + (DELTA_TAU/2) * temp;

            temp = f_B(u, B, i, idx);
            new_arr_B[v_idx[i]] = B[v_idx[i]] + (DELTA_TAU/6) * temp;
            Bk1[v_idx[i]] = B[v_idx[i]] + (DELTA_TAU/2) * temp;
        }

        // Stop
        barrier(CLK_GLOBAL_MEM_FENCE);

        FOO_VEC p1 = ^(int i, int x, int y, int z) {
            MYFLOAT ret = rk1[buffer_idx(x, y, z)];
            return ret;
        };

        temp = f_rho(rk1, uk1, idx);
        new_arr_rho[b_idx] += (DELTA_TAU/3) * temp;
        rk2[b_idx] = rho[b_idx] + (DELTA_TAU/2) * temp;
        
        for(int i = 0; i < 3; ++i) {
            temp = f_u(rk1, uk1, Bk1, p1, i, idx);
            new_arr_u[v_idx[i]] += (DELTA_TAU/3) * temp;
            uk2[v_idx[i]] = u[v_idx[i]] + (DELTA_TAU/2) * temp;

            temp = f_B(uk1, Bk1, i, idx);
            new_arr_B[v_idx[i]] += (DELTA_TAU/3) * temp;
            Bk2[v_idx[i]] = B[v_idx[i]] + (DELTA_TAU/2) * temp;
        }

        //Stop
        barrier(CLK_GLOBAL_MEM_FENCE);

        FOO_VEC p2 = ^(int i, int x, int y, int z) {
            MYFLOAT ret = rk2[buffer_idx(x, y, z)];
            return ret;
        };

        temp = f_rho(rk2, uk2, idx);
        new_arr_rho[b_idx] += (DELTA_TAU/3) * temp;
        rk3[b_idx] = rho[b_idx] + (DELTA_TAU) * temp;
        
        for(int i = 0; i < 3; ++i) {
            temp = f_u(rk2, uk2, Bk2, p2, i, idx);
            new_arr_u[v_idx[i]] += (DELTA_TAU/3) * temp;
            uk3[v_idx[i]] = u[v_idx[i]] + (DELTA_TAU) * temp;

            temp = f_B(uk2, Bk2, i, idx);
            new_arr_B[v_idx[i]] += (DELTA_TAU/3) * temp;
            Bk3[v_idx[i]] = B[v_idx[i]] + (DELTA_TAU) * temp;
        }
        
        // Stop
        barrier(CLK_GLOBAL_MEM_FENCE);

        FOO_VEC p3 = ^(int i, int x, int y, int z) {
            MYFLOAT ret = rk3[buffer_idx(x, y, z)];
            return ret;
        };

        temp = f_rho(rk3, uk3, idx);
        new_arr_rho[b_idx] += (DELTA_TAU/6) * temp;
        
        for(int i = 0; i < 3; ++i) {
            temp = f_u(rk3, uk3, Bk3, p3, i, idx);
            new_arr_u[v_idx[i]] += (DELTA_TAU/6) * temp;

            temp = f_B(uk3, Bk3, i, idx);
            new_arr_B[v_idx[i]] += (DELTA_TAU/6) * temp;
        }
}

__kernel void orzang_tang(__global MYFLOAT *u, __global MYFLOAT *B, __global MYFLOAT *rho) 
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    rho[buffer_idx(x, y, z)] = (25.0/36.0)*M_PI;

    u[vec_buffer_idx(0, x, y, z)] = -1.0f*sin(2*M_PI*y*DELTA_H_Y);
    u[vec_buffer_idx(1, x, y, z)] = sin(2*M_PI*x*DELTA_H_X);
    u[vec_buffer_idx(2, x, y, z)] = 0.0f;

    B[vec_buffer_idx(0, x, y, z)] = -1.0f*B_0*sin(2*M_PI*y*DELTA_H_Y);
    B[vec_buffer_idx(1, x, y, z)] = B_0*sin(4*M_PI*x*DELTA_H_X);
    B[vec_buffer_idx(2, x, y, z)] = 0.0f;
}