#ifndef MY_MATH
#define MY_MATH

#if TYPE == TFP32
#define MYFLOAT4 float4
#define MYFLOAT8 float8
#define MYFLOAT float
#define DISTANCE fast_distance
#else
#define MYFLOAT4 double4
#define MYFLOAT8 double8
#define MYFLOAT double
#define DISTANCE distance
#if defined(cl_khr_fp64) // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#endif

#define SIZE_X 256
#define SIZE_Y 256
#define SIZE_Z 256
#define SIZE_T 2

#define SCALAR 0
#define VECTOR 1
#define MATRIX 2

// #define DELTA_H 1e-2

#define DELTA_H ((2*3.14)/256.0)

__global const MYFLOAT8 cent_first_coef = 
    (MYFLOAT8)(1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0, 0.0, 0.0, 0.0);
__global const MYFLOAT8 cent_second_coef = 
    (MYFLOAT8)(1.0/3.0, -1.0/3.0, 0.0, -1.0/3.0, 1.0/3.0, 0.0, 0.0, 0.0);
__global const MYFLOAT8 cent_cross_coef = 
    (MYFLOAT8)(1.0/48.0, -1.0/48.0, -1.0/3.0, 1.0/3.0, 1.0/3.0, -1.0/3.0, -1.0/48.0, 1.0/48.0);

__global const MYFLOAT8 left_first_coef = 
    (MYFLOAT8)(25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0, 0.0, 0.0, 0.0);
__global const MYFLOAT8 left_second_coef = 
    (MYFLOAT8)(35.0/12.0, -26.0/3.0, 19.0/2.0, -14.0/3.0, 11.0/12.0, 0.0, 0.0, 0.0);

__global const MYFLOAT8 right_first_coef = 
    (MYFLOAT8)(-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0, 0.0, 0.0, 0.0);
__global const MYFLOAT8 right_second_coef = 
    (MYFLOAT8)(35.0/12.0, -26.0/3.0, 19.0/2.0, -14.0/3.0, 11.0/12.0, 0.0, 0.0, 0.0);

MYFLOAT my_dot(MYFLOAT8 a, MYFLOAT8 b) {
    MYFLOAT result = a.s0*b.s0 + a.s1*b.s1 + a.s2*b.s2 + a.s3*b.s3 +
        a.s4*b.s4 + a.s5*b.s5 + a.s6*b.s6 + a.s7*b.s7;
    return result;
}

int buffer_idx(int x, int y, int z) {
    return x*SIZE_Y*SIZE_Z + y*SIZE_Z + z;
}

MYFLOAT dx(MYFLOAT *A, int x, int y, int z) {
    __local MYFLOAT8 values;
    if (x > SIZE_X-3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x-1, y, z)], 
            A[buffer_idx(x-2, y, z)], A[buffer_idx(x-3, y, z)], 
            A[buffer_idx(x-4, y, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, left_first_coef)/DELTA_H;
    } else if (x < 3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x+1, y, z)], 
            A[buffer_idx(x+2, y, z)], A[buffer_idx(x+3, y, z)], 
            A[buffer_idx(x+4, y, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, right_first_coef)/DELTA_H;
    } else {
        values = (MYFLOAT8)(
            A[buffer_idx(x-2, y, z)], A[buffer_idx(x-1, y, z)], 
            A[buffer_idx(x, y, z)], A[buffer_idx(x+1, y, z)], 
            A[buffer_idx(x+2, y, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, cent_first_coef)/DELTA_H;
    }
}

MYFLOAT d2x(MYFLOAT *A, int x, int y, int z) {
    __local MYFLOAT8 values;
    if (x > SIZE_X-3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x-1, y, z)], 
            A[buffer_idx(x-2, y, z)], A[buffer_idx(x-3, y, z)], 
            A[buffer_idx(x-4, y, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, left_second_coef)/(DELTA_H*DELTA_H);
    } else if (x < 3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x+1, y, z)], 
            A[buffer_idx(x+2, y, z)], A[buffer_idx(x+3, y, z)], 
            A[buffer_idx(x+4, y, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, right_second_coef)/(DELTA_H*DELTA_H);
    } else {
        values = (MYFLOAT8)(
            A[buffer_idx(x-2, y, z)], A[buffer_idx(x-1, y, z)], 
            A[buffer_idx(x, y, z)], A[buffer_idx(x+1, y, z)], 
            A[buffer_idx(x+2, y, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, cent_second_coef)/(DELTA_H*DELTA_H);
    }
}

MYFLOAT dxdy(MYFLOAT *A, int x, int y, int z) {
    __local MYFLOAT8 values;
    if (x > SIZE_X-3) {
        return 0.0;
    } else if (x < 3) {
        return 0.0;
    } else {
        values = (MYFLOAT8)(
            A[buffer_idx(x+2, y-2, z)], A[buffer_idx(x+2, y+2, z)], 
            A[buffer_idx(x+1, y-1, z)], A[buffer_idx(x+1, y+1, z)], 
            A[buffer_idx(x-1, y-1, z)], A[buffer_idx(x-1, y+1, z)],
            A[buffer_idx(x-2, y-2, z)], A[buffer_idx(x-2, y+2, z)]);
        return my_dot(values, cent_cross_coef)/(DELTA_H*DELTA_H);
    }
}

MYFLOAT dy(MYFLOAT *A, int x, int y, int z) {
    __local MYFLOAT8 values;
    if (y > SIZE_Y-3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y-1, z)], 
            A[buffer_idx(x, y-2, z)], A[buffer_idx(x, y-3, z)], 
            A[buffer_idx(x, y-4, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, left_first_coef)/DELTA_H;
    } else if (y < 3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y+1, z)], 
            A[buffer_idx(x, y+2, z)], A[buffer_idx(x, y+3, z)], 
            A[buffer_idx(x, y+4, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, right_first_coef)/DELTA_H;
    } else {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y-2, z)], A[buffer_idx(x, y-1, z)], 
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y+1, z)], 
            A[buffer_idx(x, y+2, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, cent_first_coef)/DELTA_H;
    }
}

MYFLOAT d2y(MYFLOAT *A, int x, int y, int z) {
    __local MYFLOAT8 values;
    if (y > SIZE_Y-3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y-1, z)], 
            A[buffer_idx(x, y-2, z)], A[buffer_idx(x, y-3, z)], 
            A[buffer_idx(x, y-4, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, left_second_coef)/(DELTA_H*DELTA_H);
    } else if (y < 3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y+1, z)], 
            A[buffer_idx(x, y+2, z)], A[buffer_idx(x, y+3, z)], 
            A[buffer_idx(x, y+4, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, right_second_coef)/(DELTA_H*DELTA_H);
    } else {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y-2, z)], A[buffer_idx(x, y-1, z)], 
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y+1, z)], 
            A[buffer_idx(x, y+2, z)], 0.0,
            0.0, 0.0);
        return my_dot(values, cent_second_coef)/(DELTA_H*DELTA_H);
    }
}


MYFLOAT dz(MYFLOAT *A, int x, int y, int z) {
    __local MYFLOAT8 values;
    if (z > SIZE_Z-3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y, z-1)], 
            A[buffer_idx(x, y, z-2)], A[buffer_idx(x, y, z-3)], 
            A[buffer_idx(x, y, z-4)], 0.0,
            0.0, 0.0);
        return my_dot(values, left_first_coef)/DELTA_H;
    } else if (z < 3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y, z+1)], 
            A[buffer_idx(x, y, z+2)], A[buffer_idx(x, y, z+3)], 
            A[buffer_idx(x, y, z+4)], 0.0,
            0.0, 0.0);
        return my_dot(values, right_first_coef)/DELTA_H;
    } else {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z-2)], A[buffer_idx(x, y, z-1)], 
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y, z+1)], 
            A[buffer_idx(x, y, z+2)], 0.0,
            0.0, 0.0);
        return my_dot(values, cent_first_coef)/DELTA_H;
    }
}

MYFLOAT d2z(MYFLOAT *A, int x, int y, int z) {
    __local MYFLOAT8 values;
    if (z > SIZE_Z-3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y, z-1)], 
            A[buffer_idx(x, y, z-2)], A[buffer_idx(x, y, z-3)], 
            A[buffer_idx(x, y, z-4)], 0.0,
            0.0, 0.0);
        return my_dot(values, left_second_coef)/(DELTA_H*DELTA_H);
    } else if (z < 3) {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y, z+1)], 
            A[buffer_idx(x, y, z+2)], A[buffer_idx(x, y, z+3)], 
            A[buffer_idx(x, y, z+4)], 0.0,
            0.0, 0.0);
        return my_dot(values, right_second_coef)/(DELTA_H*DELTA_H);
    } else {
        values = (MYFLOAT8)(
            A[buffer_idx(x, y, z-2)], A[buffer_idx(x, y, z-1)], 
            A[buffer_idx(x, y, z)], A[buffer_idx(x, y, z+1)], 
            A[buffer_idx(x, y, z+2)], 0.0,
            0.0, 0.0);
        return my_dot(values, cent_second_coef)/(DELTA_H*DELTA_H);
    }
}

#endif