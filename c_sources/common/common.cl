#ifndef COMMON
#define COMMON

#include "utils_3d.cl"
#include "mhd_consts.cl"

double dxi_p(int4 i, int4 j, global double *p) {
    return kron(i, j) * dx_3D(p, get_ax(i), get_sc_idx(i), get_h(i));
}

double dxj_rho_ui_uj(int4 i, int4 j, global double *rho, global double *u) {
     char xj;
    xj = get_ax(j);
     double h;
    h = get_h(j);

     int4 rho_idx;
    rho_idx = get_sc_idx(i);

    return 
    dx_3D(rho, xj, rho_idx, h) * u[vec_buffer_idx(i)] * u[vec_buffer_idx(j)] 
        + rho[vec_buffer_idx(rho_idx)] * dx_3D(u, xj, i, h) * u[vec_buffer_idx(j)] 
        + rho[vec_buffer_idx(rho_idx)] * u[vec_buffer_idx(i)] * dx_3D(u, xj, j, h);
}

double dxi_BB(int4 i, global double *B){
     double result;
    result = 0.0;
    for (int l = 0; l < 1; ++l) {
        int4 k = (int4){l, i.s1, i.s2, i.s3};

        if (all(k == i))
            result -= dx_3D(B, get_ax(i), i, get_h(i)) * B[vec_buffer_idx(i)];
        else
            result += dx_3D(B, get_ax(i), k, get_h(i)) * B[vec_buffer_idx(k)];
        
    }
    return result;
}

double dxj_BiBj(int4 i, int4 j, global double *B) {
     char x;
    x = get_ax(j);
     double h;
    h = get_h(j);

    double result = 0.0; 

    if (all(i == j)) {
        result = dxi_BB(i, B);
    } else {
        result = - ( dx_3D(B, x, i, h) * B[vec_buffer_idx(j)] 
            + dx_3D(B, x, j, h) * B[vec_buffer_idx(i)] );
    }

    return result / (Ma*Ma);
}

double dxj_sigma_ij(int4 i, int4 j, global double *u){
    char xi;
    xi = get_ax(i);
    double hi;
    hi = get_h(i);

    char xj;
    xj = get_ax(j);
    double hj;
    hj = get_h(j);

    if (all(i == j)) {
        return (4.0/3.0)*(mu0 / Re)*(ddx_3D(u, xi, i, hi));
    } else {
        return (mu0 / Re) * (ddx_3D(u, xj, i, hj) + dxdy_3D(u, xi, xj, j, hi, hj));
    }
}

double dxj_ujBi(int4 i, int4 j, global double* u, global double* B) {
     char xj;
    xj = get_ax(j);
     double hj;
    hj = get_h(j);

    return dx_3D(u, xj, j, hj) * B[vec_buffer_idx(i)] 
        + dx_3D(B, xj, i, hj) * u[vec_buffer_idx(j)];
}

double dxj_uiBj(int4 i, int4 j, global double* u, global double* B) {
     char xj;
    xj = get_ax(j);
     double hj;
    hj = get_h(j);

    return dx_3D(u, xj, i, hj) * B[vec_buffer_idx(j)] 
        + dx_3D(B, xj, j, hj) * u[vec_buffer_idx(i)];
}

double dxj2_B(int4 i, int4 j, global double* B) {
    char xj;
    xj = get_ax(j);
    double hj;
    hj = get_h(j);

    return ddx_3D(B, xj, i, hj);
}

double dxj_rho_uj(int4 i, int4 j, global double *rho, global double *u) {
     char xj;
    xj = get_ax(j);
     double hj;
    hj = get_h(j);

     int4 rho_idx;
    rho_idx = get_sc_idx(i);

    return (dx_3D(rho, xj, rho_idx, hj)*u[vec_buffer_idx(j)] 
        + rho[vec_buffer_idx(rho_idx)]*dx_3D(u, xj, j, hj));
}

#endif