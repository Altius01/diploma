#ifndef WENO_COMMON_3D
#define WENO_COMMON_3D

#include "../common/utils_3d.cl"
#include "../common/mhd_consts.cl"

double rho_uj(int4 i, int4 j, global double *rho, global double *u) {
    int4 rho_idx = get_sc_idx(i);
    return rho[vec_buffer_idx(rho_idx)] * u[vec_buffer_idx(j)];
}

double rho_ui_uj(int4 i, int4 j, global double *rho, global double *u) {
    int4 rho_idx = get_sc_idx(i);
    return rho[vec_buffer_idx(rho_idx)] * u[vec_buffer_idx(i)] * u[vec_buffer_idx(j)];
}

double p_deltaij(int4 i, int4 j, global double *p) {
    int4 p_idx = get_sc_idx(i);
    if (i.s0 == j.s0 && i.s1 == j.s1 
        && i.s2 == j.s2 && i.s3 == j.s3) {
        return p[vec_buffer_idx(p_idx)];
        } else {
            return 0.0;
        }
}

double BiBj(int4 i, int4 j, global double *B) {
    if (i.s0 == j.s0 && i.s1 == j.s1 
        && i.s2 == j.s2 && i.s3 == j.s3) {
        double B2 = 0.0;

        for (int l = 0; l < 3; ++l) {
        int4 k = (int4){l, i.s1, i.s2, i.s3};
        B2 += ( B[vec_buffer_idx(k)] * B[vec_buffer_idx(k)] ) / (2.0);
        }
        return ( B2 - B[vec_buffer_idx(i)] * B[vec_buffer_idx(j)] ) / (Ma*Ma);
    } else {
        return -1.0*(B[vec_buffer_idx(i)] * B[vec_buffer_idx(j)] ) / (Ma*Ma);
    }
}

double uiBj_ujBi(int4 i, int4 j, global double *u, global double *B) {
    return u[vec_buffer_idx(j)]*B[vec_buffer_idx(i)] 
        - u[vec_buffer_idx(i)]*B[vec_buffer_idx(j)];
}

double sigma_ij(int4 i, int4 j, global double *u){
    char xi;
    xi = get_ax(i);
    double hi;
    hi = get_h(i);

    char xj;
    xj = get_ax(j);
    double hj;
    hj = get_h(j);

    return (1/Re) * ( dx_3D(u, xj, i, hj) + dx_3D(u, xi, j, hi) 
        - (2.0/3.0)*(dx_3D(u, xi, i, hi))*kron(i, j) );
}

double dxj_Bi(int4 i, int4 j, global double* B) {
    char xj;
    xj = get_ax(j);
    double hj;
    hj = get_h(j);

    return dx_3D(B, xj, i, hj) / Rem;
}


void get_betas(double *betas, double *arr) {
    double f0 = arr[0];
    double f1 = arr[1];
    double f2 = arr[2];
    double f3 = arr[3];
    double f4 = arr[4];

    betas[0] = (1.0/3.0 * (4.0*f0*f0 - 19.0*f0*f1 +
                       25.0*f1*f1 + 11.0*f0*f2 -
                       31.0*f1*f2 + 10.0*f2*f2));
    betas[1] = (1.0/3.0 * (4.0*f1*f1 - 13.0*f1*f2 +
                       13.0*f2*f2 + 5.0*f1*f3 -
                       13.0*f2*f3 + 4.0*f3*f3));
    betas[2] = (1.0/3.0 * (10.*f2*f2 - 31.0*f2*f3 +
                       25.0*f3*f3 + 11.0*f2*f4 -
                       19.0*f3*f4 + 4.0*f4*f4));
}

double get_omega(double g, double b) {
    double eps = 1e-6;
    return g / ( (eps+b)*(eps+b) );
}

void get_omegas(double *gammas, double *betas, double *omegas) {
    double sum_omegas = 0.0;

    for (int i = 0; i < 3; ++i) {
        omegas[i] = get_omega(gammas[i], betas[i]);
        sum_omegas += omegas[i];
    }

    for (int i = 0; i < 3; ++i) {
        omegas[i] /= sum_omegas;
    }
}

double sc_0(double *arr) {
    double f0 = arr[0];
    double f1 = arr[1];
    double f2 = arr[2];

   return 1.0/3.0 * f0 - 7.0/6.0*f1 + 11.0/6.0*f2;
}

double sc_1(double *arr) {
    double f0 = arr[1];
    double f1 = arr[2];
    double f2 = arr[3];

   return -1.0/6.0 * f0 + 5.0/6.0*f1 + 1.0/3.0*f2;
}

double sc_2(double *arr) {
    double f0 = arr[2];
    double f1 = arr[3];
    double f2 = arr[4];

   return 1.0/3.0 * f0 + 5.0/6.0*f1 - 1.0/6.0*f2;
}

double weno_solve(double *arr) {
    double gammas[3] = {0.1, 0.6, 0.3};
    double betas[3];
    get_betas(betas, arr);
    double omegas[3];
    get_omegas(gammas, betas, omegas);

    return omegas[0]*sc_0(arr) 
        + omegas[1]*sc_1(arr) 
        + omegas[2]*sc_2(arr);
}

#endif