MYFLOAT (^Diff_arr_i)(__global MYFLOAT *, int, int, int, int, int) = 
    ^(__global MYFLOAT *A, int ax, int i, int x, int y, int z) {
    MYFLOAT8 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if (ax == 0) {
        values.s0 = A[vec_buffer_idx(i, x-2, y, z)];
        values.s1 = A[vec_buffer_idx(i, x-1, y, z)];
        values.s2 = A[vec_buffer_idx(i, x, y, z)];
        values.s3 = A[vec_buffer_idx(i, x+1, y, z)];
        values.s4 = A[vec_buffer_idx(i, x+2, y, z)];
        delta = DELTA_H_X;
    } else if (ax == 1) {
        values.s0 = A[vec_buffer_idx(i, x, y-2, z)];
        values.s1 = A[vec_buffer_idx(i, x, y-1, z)];
        values.s2 = A[vec_buffer_idx(i, x, y, z)];
        values.s3 = A[vec_buffer_idx(i, x, y+1, z)];
        values.s4 = A[vec_buffer_idx(i, x, y+2, z)];
        delta = DELTA_H_Y;
    } else if (ax == 2) {
        values.s0 = A[vec_buffer_idx(i, x, y, z-2)];
        values.s1 = A[vec_buffer_idx(i, x, y, z-1)];
        values.s2 = A[vec_buffer_idx(i, x, y, z)];
        values.s3 = A[vec_buffer_idx(i, x, y, z+1)];
        values.s4 = A[vec_buffer_idx(i, x, y, z+2)];
        delta = DELTA_H_Z;
    }

    MYFLOAT ret = my_dot(values, cent_first_coef)/delta;
    return ret;
};

MYFLOAT (^Diff_arr_ii)(__global MYFLOAT *, int, int, int, int, int) = 
    ^(__global MYFLOAT *A, int ax, int i, int x, int y, int z) {
    MYFLOAT8 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if (ax == 0) {
        values.s0 = A[vec_buffer_idx(i, x-3, y, z)];
        values.s1 = A[vec_buffer_idx(i, x-2, y, z)];
        values.s2 = A[vec_buffer_idx(i, x-1, y, z)];
        values.s3 = A[vec_buffer_idx(i, x, y, z)];
        values.s4 = A[vec_buffer_idx(i, x+1, y, z)];
        values.s5 = A[vec_buffer_idx(i, x+2, y, z)];
        values.s6 = A[vec_buffer_idx(i, x+3, y, z)];
        delta = DELTA_H_X*DELTA_H_X;
    } else if (ax == 1) {
        values.s0 = A[vec_buffer_idx(i, x, y-3, z)];
        values.s1 = A[vec_buffer_idx(i, x, y-2, z)];
        values.s2 = A[vec_buffer_idx(i, x, y-1, z)];
        values.s3 = A[vec_buffer_idx(i, x, y, z)];
        values.s4 = A[vec_buffer_idx(i, x, y+1, z)];
        values.s5 = A[vec_buffer_idx(i, x, y+2, z)];
        values.s6 = A[vec_buffer_idx(i, x, y+3, z)];
        delta = DELTA_H_Y*DELTA_H_Y;
    } else if (ax == 2) {
        values.s0 = A[vec_buffer_idx(i, x, y, z-3)];
        values.s1 = A[vec_buffer_idx(i, x, y, z-2)];
        values.s2 = A[vec_buffer_idx(i, x, y, z-1)];
        values.s3 = A[vec_buffer_idx(i, x, y, z)];
        values.s4 = A[vec_buffer_idx(i, x, y, z+1)];
        values.s5 = A[vec_buffer_idx(i, x, y, z+2)];
        values.s6 = A[vec_buffer_idx(i, x, y, z+3)];
        delta = DELTA_H_Z*DELTA_H_Z;
    }

    // if (x == 255 && y == 0 && z == 0) {
    //     printf("v0:%f, v1:%f, v2:%f, v3:%f, v4:%f, v5:%f, v6:%f, v7:%f, v8%f\n",
    //         values.s0, values.s1, values.s2, values.s3, values.s4, values.s5, values.s6, values.s7);
    // }

    MYFLOAT ret = my_dot(values, cent_second_coef)/delta;
    return ret;
};

MYFLOAT (^Diff_arr_ij)(__global MYFLOAT *, int, int, int, int, int, int) = 
    ^(__global MYFLOAT *A, int ax_i, int ax_j, int i, int x, int y, int z) {
    MYFLOAT16 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if ((ax_i == 0 && ax_j == 1) || (ax_i == 1 && ax_j == 0)) {
        values.s0 = A[vec_buffer_idx(i, x-2, y-2, z)];
        values.s1 = A[vec_buffer_idx(i, x-1, y-1, z)];
        values.s2 = A[vec_buffer_idx(i, x-1, y+0, z)];
        values.s3 = A[vec_buffer_idx(i, x+1, y+1, z)];
        values.s4 = A[vec_buffer_idx(i, x+0, y-1, z)];
        values.s5 = A[vec_buffer_idx(i, x+0, y+1, z)];
        values.s6 = A[vec_buffer_idx(i, x+1, y-1, z)];
        values.s7 = A[vec_buffer_idx(i, x+1, y+0, z)];
        values.s8 = A[vec_buffer_idx(i, x+1, y+1, z)];
        values.s9 = A[vec_buffer_idx(i, x+2, y+2, z)];
        delta = DELTA_H_X*DELTA_H_Y;
    } else if ((ax_i == 0 && ax_j == 2) || (ax_i == 2 && ax_j == 0)) {
        values.s0 = A[vec_buffer_idx(i, x-2, y, z-2)];
        values.s1 = A[vec_buffer_idx(i, x-1, y, z-1)];
        values.s2 = A[vec_buffer_idx(i, x-1, y, z+0)];
        values.s3 = A[vec_buffer_idx(i, x+1, y, z+1)];
        values.s4 = A[vec_buffer_idx(i, x+0, y, z-1)];
        values.s5 = A[vec_buffer_idx(i, x+0, y, z+1)];
        values.s6 = A[vec_buffer_idx(i, x+1, y, z-1)];
        values.s7 = A[vec_buffer_idx(i, x+1, y, z+0)];
        values.s8 = A[vec_buffer_idx(i, x+1, y, z+1)];
        values.s9 = A[vec_buffer_idx(i, x+2, y, z+2)];
        delta = DELTA_H_X*DELTA_H_Z;
    } else if ((ax_i == 2 && ax_j == 1) || (ax_i == 1 && ax_j == 2)) {
        values.s0 = A[vec_buffer_idx(i, x, y-2, z-2)];
        values.s1 = A[vec_buffer_idx(i, x, y-1, z-1)];
        values.s2 = A[vec_buffer_idx(i, x, y+0, z-1)];
        values.s3 = A[vec_buffer_idx(i, x, y+1, z+1)];
        values.s4 = A[vec_buffer_idx(i, x, y-1, z+0)];
        values.s5 = A[vec_buffer_idx(i, x, y+1, z+0)];
        values.s6 = A[vec_buffer_idx(i, x, y-1, z+1)];
        values.s7 = A[vec_buffer_idx(i, x, y+0, z+1)];
        values.s8 = A[vec_buffer_idx(i, x, y+1, z+1)];
        values.s9 = A[vec_buffer_idx(i, x, y+2, z+2)];
        delta = DELTA_H_Z*DELTA_H_Y;
    }

    MYFLOAT ret = my_dot_16(values, cent_cross_coef)/delta;
    return ret;
};