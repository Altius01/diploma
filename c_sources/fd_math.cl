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

#define L 2*2*M_PI
#define DELTA_H_X (L/(SIZE_X))
#define DELTA_H_Y (L/(SIZE_Y))
#define DELTA_H_Z (L/(SIZE_Z))

MYFLOAT MU = 10.0f;
MYFLOAT RE = 1.0f;
MYFLOAT MA = 1.0f;
MYFLOAT REM = 1.0f;
MYFLOAT GAMMA = 5/3.0f;


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

typedef MYFLOAT (^FOO)(int, int, int);
typedef FOO (^FOO_VEC)(int);
typedef MYFLOAT (^SIGMA)(FOO_VEC, int, int, int, int ,int);

//MYFLOAT (^FOO)(MYFLOAT*, int, int, int)

MYFLOAT (^Diff_i)(FOO, int, int, int, int) = ^(FOO A, int ax, int x, int y, int z) {
    MYFLOAT8 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if (ax == 0) {
        values.s0 = A(x-2, y, z);
        values.s1 = A(x-1, y, z);
        values.s2 = A(x, y, z);
        values.s3 = A(x+1, y, z);
        values.s4 = A(x+2, y, z);
        delta = DELTA_H_X;
    } else if (ax == 1) {
        values.s0 = A(x, y-2, z);
        values.s1 = A(x, y-1, z);
        values.s2 = A(x, y, z);
        values.s3 = A(x, y+1, z);
        values.s4 = A(x, y+2, z);
        delta = DELTA_H_Y;
    } else if (ax == 2) {
        values.s0 = A(x, y, z-2);
        values.s1 = A(x, y, z-1);
        values.s2 = A(x, y, z);
        values.s3 = A(x, y, z+1);
        values.s4 = A(x, y, z+2);
        delta = DELTA_H_Z;
    }

    MYFLOAT ret = my_dot(values, cent_first_coef)/delta;
    return ret;
};

MYFLOAT (^Diff_ii)(FOO, int, int, int, int) = ^(FOO A, int ax, int x, int y, int z) {
    MYFLOAT8 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if (ax == 0) {
        values.s0 = A(x-3, y, z);
        values.s1 = A(x-2, y, z);
        values.s2 = A(x-1, y, z);
        values.s3 = A(x, y, z);
        values.s4 = A(x+1, y, z);
        values.s5 = A(x+2, y, z);
        values.s6 = A(x+3, y, z);
        delta = DELTA_H_X*DELTA_H_X;
    } else if (ax == 1) {
        values.s0 = A(x, y-3, z);
        values.s1 = A(x, y-2, z);
        values.s2 = A(x, y-1, z);
        values.s3 = A(x, y, z);
        values.s4 = A(x, y+1, z);
        values.s5 = A(x, y+2, z);
        values.s6 = A(x, y+3, z);
        delta = DELTA_H_Y*DELTA_H_Y;
    } else if (ax == 2) {
        values.s0 = A(x, y, z-3);
        values.s1 = A(x, y, z-2);
        values.s2 = A(x, y, z-1);
        values.s3 = A(x, y, z);
        values.s4 = A(x, y, z+1);
        values.s5 = A(x, y, z+2);
        values.s6 = A(x, y, z+3);
        delta = DELTA_H_Z*DELTA_H_Z;
    }

    if (x == 255 && y == 0 && z == 0) {
        printf("v0:%f, v1:%f, v2:%f, v3:%f, v4:%f, v5:%f, v6:%f, v7:%f, v8%f\n",
            values.s0, values.s1, values.s2, values.s3, values.s4, values.s5, values.s6, values.s7);
    }

    MYFLOAT ret = my_dot(values, cent_second_coef)/delta;
    return ret;
};

MYFLOAT (^Diff_ij)(FOO, int, int, int, int, int) = ^(FOO A, int ax_i, int ax_j, int x, int y, int z) {
    MYFLOAT16 values = (0.0f);
    MYFLOAT delta = 1.0f;
    if ((ax_i == 0 && ax_j == 1) || (ax_i == 1 && ax_j == 0)) {
        values.s0 = A(x-2, y-2, z);
        values.s1 = A(x-1, y-1, z);
        values.s2 = A(x-1, y+0, z);
        values.s3 = A(x+1, y+1, z);
        values.s4 = A(x+0, y-1, z);
        values.s5 = A(x+0, y+1, z);
        values.s6 = A(x+1, y-1, z);
        values.s7 = A(x+1, y+0, z);
        values.s8 = A(x+1, y+1, z);
        values.s9 = A(x+2, y+2, z);
        delta = DELTA_H_X*DELTA_H_Y;
    } else if ((ax_i == 0 && ax_j == 2) || (ax_i == 2 && ax_j == 0)) {
        values.s0 = A(x-2, y, z-2);
        values.s1 = A(x-1, y, z-1);
        values.s2 = A(x-1, y, z+0);
        values.s3 = A(x+1, y, z+1);
        values.s4 = A(x+0, y, z-1);
        values.s5 = A(x+0, y, z+1);
        values.s6 = A(x+1, y, z-1);
        values.s7 = A(x+1, y, z+0);
        values.s8 = A(x+1, y, z+1);
        values.s9 = A(x+2, y, z+2);
        delta = DELTA_H_X*DELTA_H_Z;
    } else if ((ax_i == 2 && ax_j == 1) || (ax_i == 1 && ax_j == 2)) {
        values.s0 = A(x, y-2, z-2);
        values.s1 = A(x, y-1, z-1);
        values.s2 = A(x, y+0, z-1);
        values.s3 = A(x, y+1, z+1);
        values.s4 = A(x, y-1, z+0);
        values.s5 = A(x, y+1, z+0);
        values.s6 = A(x, y-1, z+1);
        values.s7 = A(x, y+0, z+1);
        values.s8 = A(x, y+1, z+1);
        values.s9 = A(x, y+2, z+2);
        delta = DELTA_H_Z*DELTA_H_Y;
    }
    
    MYFLOAT ret = my_dot_16(values, cent_cross_coef)/delta;
    return ret;
};

SIGMA s = ^(FOO_VEC u, int i, int j, int x, int y, int z) {
    MYFLOAT res = 0.5*(Diff_i(u(i), j, x, y, z) + Diff_i(u(j), i, x, y, z));
    return res;
};

SIGMA sigma = ^(FOO_VEC u, int i, int j, int x, int y, int z) {
    MYFLOAT res = MU*(2.0*s(u, i, j, x, y, z) - (2.0/3.0)*s(u, i, j, x, y, z)*kron(i, j));
    return res;
};

FOO f_rho(FOO rho, FOO_VEC u, int j, int x, int y, int z) {
    FOO ret = ^(int x, int y, int z) {
        return rho(x, y, z)*u(j)(x, y, z);
    };
}

__kernel void test_inital_data(__global MYFLOAT *foo, __global MYFLOAT *der)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    foo[buffer_idx(x, y, z)] = cos(z*DELTA_H_X)*(1.0+sin(y*DELTA_H_Y));
    der[buffer_idx(x, y, z)] = -1.0*sin(z*DELTA_H_X)*cos(y*DELTA_H_Y);
}

__kernel void test_first_der_x(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_i(arr, 0, x, y, z);
}

__kernel void test_first_der_y(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_i(arr, 1, x, y, z);
}

__kernel void test_first_der_z(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_i(arr, 2, x, y, z);
}

__kernel void test_second_der_x(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_ii(arr, 0, x, y, z);
}

__kernel void test_second_der_y(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_ii(arr, 1, x, y, z);
}

__kernel void test_second_der_z(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_ii(arr, 2, x, y, z);
}

__kernel void test_dxdy(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_ij(arr, 0, 1, x, y, z);
}

__kernel void test_dxdz(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_ij(arr, 0, 2, x, y, z);
}

__kernel void test_dydz(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    FOO arr = ^(int x, int y, int z) {
        MYFLOAT ret = foo[buffer_idx(x, y, z)];
        return ret;
    };
    res[buffer_idx(x, y, z)] = Diff_ij(arr, 1, 2, x, y, z);
}

#endif

