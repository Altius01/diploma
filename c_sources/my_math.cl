#ifndef TESTS
#define TESTS

#include "fd_math.cl"

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