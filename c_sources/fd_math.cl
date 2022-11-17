#ifndef FD_MATH
#define FD_MATH

#if TYPE == TFP32
#define MYFLOAT4 float4
#define MYFLOAT8 float8
#define MYFLOAT16 float16
#define MYFLOAT float
#define DISTANCE fast_distance
#else
#define MYFLOAT4 double4
#define MYFLOAT8 double8
#define MYFLOAT16 double16
#define MYFLOAT double
#define DISTANCE distance
#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#endif

#define SIZE_X 256
#define SIZE_Y 256
#define SIZE_Z 256

#define L 1.0f //2*2*M_PI
#define DELTA_H_X (L/(SIZE_X))
#define DELTA_H_Y (L/(SIZE_Y))
#define DELTA_H_Z (L/(SIZE_Z))

#define DELTA_TAU 1e-7

MYFLOAT MU = 10.0f;
MYFLOAT RE = 1.0f;
MYFLOAT MA = 1.0f;
MYFLOAT REM = 1.0f;
MYFLOAT GAMMA = 5/3.0f;

MYFLOAT B_0 = 0.282094f;


__global const MYFLOAT8 cent_first_coef = 
    (MYFLOAT8)(1.0/12.0, -8.0/12.0, 0.0, 8.0/12.0, -1.0/12.0, 0.0, 0.0, 0.0);
__global const MYFLOAT8 cent_second_coef = 
    (MYFLOAT8) (0.0, 1.0/3, -1.0/3, 0.0, -1.0/3, 1.0/3,0.0, 0.0);
// __global const MYFLOAT8 cent_cross_coef = 
//     (MYFLOAT8)(1.0/48.0, -1.0/48.0, -1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0/3.0, -1.0/48.0, 1.0/48.0);

__global const MYFLOAT16 cent_cross_coef =
(MYFLOAT16) (-0.16666666666666696, 1.166666666666668,
-0.5000000000000009, 1.1666666666666659,
-0.5000000000000006, -0.4999999999999994,
0.0, -0.49999999999999944,
0.0, -0.16666666666666655,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

MYFLOAT my_dot(MYFLOAT8 a, MYFLOAT8 b) {
    MYFLOAT result = a.s0*b.s0 + a.s1*b.s1 + a.s2*b.s2 + a.s3*b.s3 +
        a.s4*b.s4 + a.s5*b.s5 + a.s6*b.s6 + a.s7*b.s7;
    return result;
}

MYFLOAT my_dot_16(MYFLOAT16 a, MYFLOAT16 b) {
    MYFLOAT result = a.s0*b.s0 + a.s1*b.s1 + a.s2*b.s2 + a.s3*b.s3 +
        a.s4*b.s4 + a.s5*b.s5 + a.s6*b.s6 + a.s7*b.s7 + 
        a.s8*b.s8 + a.s9*b.s9 + a.sA*b.sA + a.sB*b.sB +
        a.sC*b.sC + a.sD*b.sD + a.sE*b.sE + a.sF*b.sF;
    return result;
}

MYFLOAT kron(int i, int j) {
    return i == j ? 1.0f : 0.0f;
}

int buffer_idx(int x, int y, int z) {
    if (x >= SIZE_X)
        x = x%SIZE_X;
    else if (x < 0)
        x += SIZE_X;

    if (y >= SIZE_Y)
        y = y%SIZE_Y;
    else if (y < 0)
        y += SIZE_Y;

    if (z >= SIZE_Z)
        z = z%SIZE_Z;
    else if (z < 0)
        z += SIZE_Z;

    return x*SIZE_Y*SIZE_Z + y*SIZE_Z + z;
}

int vec_buffer_idx(int i, int x, int y, int z) {
    if (x >= SIZE_X)
        x = x%SIZE_X;
    else if (x < 0)
        x += SIZE_X;

    if (y >= SIZE_Y)
        y = y%SIZE_Y;
    else if (y < 0)
        y += SIZE_Y;

    if (z >= SIZE_Z)
        z = z%SIZE_Z;
    else if (z < 0)
        z += SIZE_Z;

    // if (i*SIZE_X*SIZE_Y*SIZE_Z + x*SIZE_Y*SIZE_Z + y*SIZE_Z + z > 3*SIZE_X*SIZE_Y*SIZE_Z)
    //     printf("\nAlarm! i:%d\n", i);

    return i*SIZE_X*SIZE_Y*SIZE_Z + x*SIZE_Y*SIZE_Z + y*SIZE_Z + z;
}

typedef MYFLOAT (^FOO)(int, int, int);
typedef MYFLOAT (^FOO_VEC)(int, int, int, int);
typedef MYFLOAT (^SIGMA)(FOO_VEC, int, int, int, int ,int);
typedef MYFLOAT (^SIGMA_ARR)(__global MYFLOAT *, int, int, int, int ,int);

//MYFLOAT (^FOO)(MYFLOAT*, int, int, int)

MYFLOAT (^Diff_i)(FOO_VEC, int, int, int, int, int) = ^(FOO_VEC A, int ax, int i, int x, int y, int z) {
    MYFLOAT8 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if (ax == 0) {
        values.s0 = A(i, x-2, y, z);
        values.s1 = A(i, x-1, y, z);
        values.s2 = A(i, x, y, z);
        values.s3 = A(i, x+1, y, z);
        values.s4 = A(i, x+2, y, z);
        delta = DELTA_H_X;
    } else if (ax == 1) {
        values.s0 = A(i, x, y-2, z);
        values.s1 = A(i, x, y-1, z);
        values.s2 = A(i, x, y, z);
        values.s3 = A(i, x, y+1, z);
        values.s4 = A(i, x, y+2, z);
        delta = DELTA_H_Y;
    } else if (ax == 2) {
        values.s0 = A(i, x, y, z-2);
        values.s1 = A(i, x, y, z-1);
        values.s2 = A(i, x, y, z);
        values.s3 = A(i, x, y, z+1);
        values.s4 = A(i, x, y, z+2);
        delta = DELTA_H_Z;
    }

    MYFLOAT ret = my_dot(values, cent_first_coef)/delta;
    return ret;
};

MYFLOAT (^Diff_ii)(FOO_VEC, int, int, int, int, int) = ^(FOO_VEC A, int ax, int i, int x, int y, int z) {
    MYFLOAT8 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if (ax == 0) {
        values.s0 = A(i, x-3, y, z);
        values.s1 = A(i, x-2, y, z);
        values.s2 = A(i, x-1, y, z);
        values.s3 = A(i, x, y, z);
        values.s4 = A(i, x+1, y, z);
        values.s5 = A(i, x+2, y, z);
        values.s6 = A(i, x+3, y, z);
        delta = DELTA_H_X*DELTA_H_X;
    } else if (ax == 1) {
        values.s0 = A(i, x, y-3, z);
        values.s1 = A(i, x, y-2, z);
        values.s2 = A(i, x, y-1, z);
        values.s3 = A(i, x, y, z);
        values.s4 = A(i, x, y+1, z);
        values.s5 = A(i, x, y+2, z);
        values.s6 = A(i, x, y+3, z);
        delta = DELTA_H_Y*DELTA_H_Y;
    } else if (ax == 2) {
        values.s0 = A(i, x, y, z-3);
        values.s1 = A(i, x, y, z-2);
        values.s2 = A(i, x, y, z-1);
        values.s3 = A(i, x, y, z);
        values.s4 = A(i, x, y, z+1);
        values.s5 = A(i, x, y, z+2);
        values.s6 = A(i, x, y, z+3);
        delta = DELTA_H_Z*DELTA_H_Z;
    }

    // if (x == 255 && y == 0 && z == 0) {
    //     printf("v0:%f, v1:%f, v2:%f, v3:%f, v4:%f, v5:%f, v6:%f, v7:%f, v8%f\n",
    //         values.s0, values.s1, values.s2, values.s3, values.s4, values.s5, values.s6, values.s7);
    // }

    MYFLOAT ret = my_dot(values, cent_second_coef)/delta;
    return ret;
};

MYFLOAT (^Diff_ij)(FOO_VEC, int, int, int, int, int, int) = ^(FOO_VEC A, int ax_i, int ax_j, int i, int x, int y, int z) {
    MYFLOAT16 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if ((ax_i == 0 && ax_j == 1) || (ax_i == 1 && ax_j == 0)) {
        values.s0 = A(i, x-2, y-2, z);
        values.s1 = A(i, x-1, y-1, z);
        values.s2 = A(i, x-1, y+0, z);
        values.s3 = A(i, x+1, y+1, z);
        values.s4 = A(i, x+0, y-1, z);
        values.s5 = A(i, x+0, y+1, z);
        values.s6 = A(i, x+1, y-1, z);
        values.s7 = A(i, x+1, y+0, z);
        values.s8 = A(i, x+1, y+1, z);
        values.s9 = A(i, x+2, y+2, z);
        delta = DELTA_H_X*DELTA_H_Y;
    } else if ((ax_i == 0 && ax_j == 2) || (ax_i == 2 && ax_j == 0)) {
        values.s0 = A(i, x-2, y, z-2);
        values.s1 = A(i, x-1, y, z-1);
        values.s2 = A(i, x-1, y, z+0);
        values.s3 = A(i, x+1, y, z+1);
        values.s4 = A(i, x+0, y, z-1);
        values.s5 = A(i, x+0, y, z+1);
        values.s6 = A(i, x+1, y, z-1);
        values.s7 = A(i, x+1, y, z+0);
        values.s8 = A(i, x+1, y, z+1);
        values.s9 = A(i, x+2, y, z+2);
        delta = DELTA_H_X*DELTA_H_Z;
    } else if ((ax_i == 2 && ax_j == 1) || (ax_i == 1 && ax_j == 2)) {
        values.s0 = A(i, x, y-2, z-2);
        values.s1 = A(i, x, y-1, z-1);
        values.s2 = A(i, x, y+0, z-1);
        values.s3 = A(i, x, y+1, z+1);
        values.s4 = A(i, x, y-1, z+0);
        values.s5 = A(i, x, y+1, z+0);
        values.s6 = A(i, x, y-1, z+1);
        values.s7 = A(i, x, y+0, z+1);
        values.s8 = A(i, x, y+1, z+1);
        values.s9 = A(i, x, y+2, z+2);
        delta = DELTA_H_Z*DELTA_H_Y;
    }

    MYFLOAT ret = my_dot_16(values, cent_cross_coef)/delta;
    return ret;
};

MYFLOAT (^Diff_arr_i)(__global MYFLOAT *, int, int, int, int, int) = 
    ^(__global MYFLOAT *A, int ax, int i, int x, int y, int z) {
    MYFLOAT8 values = (0.0f);
    MYFLOAT delta = 1.0f;

    if(i > 2)
        printf("\nDiff_i\n");

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

    // if(i > 2)
    //     printf("\nDiff_ii\n");

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

    if(i > 2)
        printf("\nDiff_ij\n");

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

MYFLOAT prod2_diff_i(FOO_VEC a, FOO_VEC b, int3 axes, int3 idx) {
    return Diff_i(a, axes.x, axes.y, idx.x, idx.y, idx.z) * b(axes.z, idx.x, idx.y, idx.z) +
        Diff_i(b, axes.x, axes.z, idx.x, idx.y, idx.z) * a(axes.y, idx.x, idx.y, idx.z);
}

MYFLOAT prod3_diff_i(FOO_VEC a, FOO_VEC b, FOO_VEC c, int4 axes, int3 idx) {
    return Diff_i(a, axes.x, axes.y, idx.x, idx.y, idx.z) *
                b(axes.z, idx.x, idx.y, idx.z) * c(axes.s3, idx.x, idx.y, idx.z) +
        Diff_i(b, axes.x, axes.z, idx.x, idx.y, idx.z) * 
                a(axes.y, idx.x, idx.y, idx.z)*c(axes.s3, idx.x, idx.y, idx.z) +
        Diff_i(c, axes.x, axes.s3, idx.x, idx.y, idx.z) * 
                a(axes.y, idx.x, idx.y, idx.z) * b(axes.z, idx.x, idx.y, idx.z);
}

MYFLOAT prod2_arr_diff_i(__global MYFLOAT *a, __global MYFLOAT *b, int3 axes, int3 idx) {
    return Diff_arr_i(a, axes.x, axes.y, idx.x, idx.y, idx.z) * 
            b[vec_buffer_idx(axes.z, idx.x, idx.y, idx.z)] +
        Diff_arr_i(b, axes.x, axes.z, idx.x, idx.y, idx.z) * 
            a[vec_buffer_idx(axes.y, idx.x, idx.y, idx.z)];
}

MYFLOAT prod3_arr_diff_i(__global MYFLOAT *a, __global MYFLOAT *b, __global MYFLOAT *c, int4 axes, int3 idx) {
    return Diff_arr_i(a, axes.x, axes.y, idx.x, idx.y, idx.z) *
                b[vec_buffer_idx(axes.z, idx.x, idx.y, idx.z)] * 
                c[vec_buffer_idx(axes.s3, idx.x, idx.y, idx.z)] +
        Diff_arr_i(b, axes.x, axes.z, idx.x, idx.y, idx.z) * 
                a[vec_buffer_idx(axes.y, idx.x, idx.y, idx.z)]*
                c[vec_buffer_idx(axes.s3, idx.x, idx.y, idx.z)] +
        Diff_arr_i(c, axes.x, axes.s3, idx.x, idx.y, idx.z) * 
                a[vec_buffer_idx(axes.y, idx.x, idx.y, idx.z)] * 
                b[vec_buffer_idx(axes.z, idx.x, idx.y, idx.z)];
}

MYFLOAT abs_prod_diff(FOO_VEC a, int ax, int3 idx) {
    MYFLOAT result = 0.0f;
    int3 axes = 0;
    for(int i = 0; i < 3; ++i) {
        axes.y = i; axes.z = i;
        result += prod2_diff_i(a, a, axes, idx);
    }
    return result;
}

MYFLOAT abs_prod_arr_diff(__global MYFLOAT *a, int ax, int3 idx) {
    MYFLOAT result = 0.0f;
    int3 axes = 0;
    for(int i = 0; i < 3; ++i) {
        axes.y = i; axes.z = i;
        result += prod2_arr_diff_i(a, a, axes, idx);
    }
    return result;
}

SIGMA s = ^(FOO_VEC u, int i, int j, int x, int y, int z) {
    MYFLOAT res = 0.5*(Diff_i(u, j, i, x, y, z) + Diff_i(u, i, j, x, y, z));
    return res;
};

SIGMA s_second = ^(FOO_VEC u, int i, int j, int x, int y, int z) {
    MYFLOAT res = 0.5*(Diff_ii(u, j, i, x, y, z) + Diff_ij(u, i, j, j, x, y, z));
    return res;
};

SIGMA_ARR s_second_arr = ^(__global MYFLOAT *u, int i, int j, int x, int y, int z) {
    if (i > 2)
        printf("\nSIGMA i:%d\n", i);
    MYFLOAT res = 0.5*(Diff_arr_ii(u, j, i, x, y, z) + Diff_arr_ij(u, i, j, j, x, y, z));
    return res;
};

MYFLOAT sigma(__global MYFLOAT *u, SIGMA_ARR s, int i, int j, int x, int y, int z) {
    MYFLOAT res = MU*(2.0*s(u, i, j, x, y, z) - (2.0/3.0)*s(u, i, j, x, y, z)*kron(i, j));
    return res;
}

MYFLOAT vec_abs(__global MYFLOAT *a, int x, int y, int z) {
    MYFLOAT result = 0.0f;
    for (int i = 0; i < 3; ++i) {
        result += a[vec_buffer_idx(i, x, y, z)] * a[vec_buffer_idx(i, x, y, z)];
    }
    return result;
}
#endif

