#ifndef FD_FLUX_3D
#define FD_FLUX_3D

#include "..\common\common.cl"

double flux_rho(int4 i, global double* rho, global double* u) {
     double result;
    result = 0.0;

     int4 j;
    for (int k = 0; k < 3; ++k) {
        j = (int4) {k, i.s1, i.s2, i.s3};
        result += dxj_rho_uj(i, j, rho, u);
    }
    return result;
}

double flux_u(
    int4 i, 
    global double *u, global double *B,
    global double *rho, global double *p
) {
     double result;
    result = 0.0;

     int4 j;
    for (int k = 0; k < 3; ++k) {
        j = (int4) {k, i.s1, i.s2, i.s3};
        result += dxj_rho_ui_uj(i, j, rho, u) + dxi_p(i, j, p) + dxj_BiBj(i, j, B);
    }
    return result;
}

double diff_u(
    int4 i, 
    global double *u
) {
    double result;
    result = 0.0;

    int4 j;
    for (int k = 0; k < 3; ++k) {
        j = (int4) {k, i.s1, i.s2, i.s3};
        result += dxj_sigma_ij(i, j, u);
    }

    return result;
}

double flux_B(int4 i, global double *u, global double *B) {
     double result;
    result = 0.0;

     int4 j;
    for (int k = 0; k < 3; ++k) {
        j = (int4) {k, i.s1, i.s2, i.s3};
        result += dxj_ujBi(i, j, u, B) - dxj_uiBj(i, j, u, B);
    }

    return result;
}

double diff_B(int4 i, global double *B) {
     double result;
    result = 0.0;

     int4 j;
    for (int k = 0; k < 3; ++k) {
        j = (int4) {k, i.s1, i.s2, i.s3};
        result += dxj2_B(i, j, B);
    }

    return result / Rem;
}

#endif