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
#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#endif

#define DELTA_TAU 1e-5

#define MU 10
#define RE 1
#define MA 1
#define REM 1
#define GAMMA 5/3

#import "my_math.cl"

#define KRON(i, j) (i==j) ? 1.0f : 0.0f

__kernel void test_inital_data(__global MYFLOAT *foo, __global MYFLOAT *der)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    foo[buffer_idx(x, y, z)] = cos(x*DELTA_H)*sin(y*DELTA_H) + cos(z*DELTA_H);
    der[buffer_idx(x, y, z)] = 0.0;
}

__kernel void test_first_der_x(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    res[buffer_idx(x, y, z)] = dx(foo, x, y, z);
}

__kernel void test_first_der_y(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    res[buffer_idx(x, y, z)] = dy(foo, x, y, z);
}

__kernel void test_first_der_z(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    res[buffer_idx(x, y, z)] = dz(foo, x, y, z);
}

__kernel void test_second_der_x(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    res[buffer_idx(x, y, z)] = d2x(foo, x, y, z);
}

__kernel void test_second_der_y(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    res[buffer_idx(x, y, z)] = d2y(foo, x, y, z);
}

__kernel void test_second_der_z(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    res[buffer_idx(x, y, z)] = d2z(foo, x, y, z);
}

__kernel void test_dxdy(__global MYFLOAT *foo, __global MYFLOAT *res)
{
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);
    res[buffer_idx(x, y, z)] = dxdy(foo, x, y, z);
}