#include "common\mhd_consts.cl"
#include "common\common.cl"
#include "common\utils_3d.cl"
#include "common\ghost_cells.cl"

__kernel void test_dx_mul(global double* a, global double* b, global double* c, global double* ans) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 idx = (int4) {0, x, y, z};

    int4 idxs[3] = {idx, idx, idx};
    global double* arrs[3] = {a, b, c};

    ans[vec_buffer_idx(idx)] = dx_mul_3D(3, 'x', char_h[0], idxs, arrs);
}

__kernel void test_div(global double* vec, global double* ans) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int4 idx = (int4) {0, x, y, z};

    ans[vec_buffer_idx(idx)] = div(idx, vec);
}

__kernel void test_rot(global double* vec, global double* ans) {
    int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

    int4 idx = (int4) {0, x, y, z};
    int4 idxs[3];
    get_indxs(idx, idxs);

    double3 rotor = rot(idx, vec);
    ans[vec_buffer_idx(idxs[0])] = rotor.x;
    ans[vec_buffer_idx(idxs[1])] = rotor.y;
    ans[vec_buffer_idx(idxs[2])] = rotor.z;
}
