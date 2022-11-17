#include "common\utils_3d.cl"

__kernel void test_dx(global double* arr, global double *arr_diff) {
    int x = get_global_id(0) + 2;
    double hx = 2.0*M_PI*1.0/(get_global_size(0)-4);

    int4 xid = (int4) {0, 0, 0, x};

    printf("%lf\n", hx);
    arr_diff[x] = dx_3D(arr, 'z', xid, hx);
}

__kernel void test_boundaries(global double *arr) {
    int x = get_global_id(0);
    int N = get_global_size(0) - 4;
    int new_x = x;
    
    if (x < 2) {
        new_x += N - 4;
    } else if (x >= N + 2) {
        new_x -= N - 4;
    }
    
    arr[x] = arr[new_x];
}