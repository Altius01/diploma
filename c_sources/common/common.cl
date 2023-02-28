#ifndef COMMON
#define COMMON

#include "utils_3d.cl"
#include "mhd_consts.cl"

double div(int4 i, global double *arr) {
    int4 idxs[3];
    get_indxs(i, idxs);

    return dx_3D(arr, 'x', idxs[0], get_h(0))
        + dx_3D(arr, 'y', idxs[1], get_h(1)) 
        + dx_3D(arr, 'z', idxs[2], get_h(2));
}

double3 rot(int4 i, global double *arr) {
    double result[3] = {0, 0, 0}; 
    double3 rotor;
    int4 idx[3];
    get_indxs(i, idx);

    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            for (char m = 0; m < 3; ++m) {
                result[k] += LEVI_CIVITA[k][l][m] * dx_3D(arr, char_ax[l], idx[m], char_h[l]);
            }
        }
    }

    // rotor.x = dx_3D(arr, 'y', idx[2], get_h(1)) - dx_3D(arr, 'z', idx[1], get_h(2));
    // rotor.y = dx_3D(arr, 'z', idx[0], get_h(2)) - dx_3D(arr, 'x', idx[2], get_h(0));
    // rotor.z = dx_3D(arr, 'x', idx[1], get_h(0)) - dx_3D(arr, 'y', idx[0], get_h(1));

    rotor.x = result[0];
    rotor.y = result[1];
    rotor.z = result[2];

    return rotor;
}

double dxi_p(int4 i, int4 j, global double *p) {
    return kron(i, j) * dx_3D(p, get_ax(i), get_sc_idx(i), get_h(i));
}

double dxj_rho_ui_uj(int4 i, int4 j, global double *rho, global double *u) {
    char xj;
    xj = get_ax(j);
    double h;
    h = get_h(j);

    // int4 rho_idx;
    // rho_idx = get_sc_idx(i);

    // // return 
    // double ret = dx_3D(rho, xj, rho_idx, h) * u[vec_buffer_idx(i)] * u[vec_buffer_idx(j)] 
    //     + rho[vec_buffer_idx(rho_idx)] * dx_3D(u, xj, i, h) * u[vec_buffer_idx(j)] 
    //     + rho[vec_buffer_idx(rho_idx)] * u[vec_buffer_idx(i)] * dx_3D(u, xj, j, h);

    int4 idx[3] = {get_sc_idx(i), i, j};
    global double* arrs[3] = {rho, u, u};

    // if (fabs(ret - dx_mul_3D(3, xj, h, idx, arrs)) > pow(get_h(0), 3))
    //     printf("ERROR: %lf\n", fabs(ret - dx_mul_3D(3, xj, h, idx, arrs)));
    return dx_mul_3D(3, xj, h, idx, arrs);
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
    // char x;
    // x = get_ax(j);
    // double h;
    // h = get_h(j);

    // double old_result = 0.0; 

    // if (all(i == j)) {
    //     old_result = dxi_BB(i, B);
    // } else {
    //     old_result = - ( dx_3D(B, x, i, h) * B[vec_buffer_idx(j)] 
    //         + dx_3D(B, x, j, h) * B[vec_buffer_idx(i)] );
    // }

    // return old_result / (Ma*Ma);

    int4 idxs[3];
    get_indxs(i, idxs);
    double result = 0;
    
    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            for (char m = 0; m < 3; ++m) {
                result -= levi_civita(i, j, idxs[k]) * levi_civita(j, idxs[l], idxs[m]) 
                    * B[vec_buffer_idx(idxs[k])] * dx_3D(B, char_ax[l], idxs[m], char_h[l]);
            }
        }
    }

    // if (fabs(old_result - result) > pow(get_h(0), 3))
    //     printf("ERROR: %lf!\n", fabs(old_result - result));
    return result / (Ma*Ma);
}

double S_ij(int4 i, int4 j, global double* u) {
    return 0.5 * (dx_3D(u, get_ax(j), i, get_h(j)) + dx_3D(u, get_ax(i), j, get_h(i)));
}

double dxj_S_ij(int4 i, int4 j, global double* u) {
    return 0.5 * (ddx_3D(u, get_ax(j), i, get_h(j)) + dxdy_3D(u, get_ax(i), get_ax(j), j, get_h(i), get_h(j)));
}

double dxj_sigma_ij(int4 i, int4 j, global double* u){
    // char xi;
    // xi = get_ax(i);
    // double hi;
    // hi = get_h(i);

    // char xj;
    // xj = get_ax(j);
    // double hj;
    // hj = get_h(j);

    // if (all(i == j)) {
    //     return (4.0/3.0)*(mu0 / Re)*(ddx_3D(u, xi, i, hi));
    // } else {
    //     return (mu0 / Re) * (ddx_3D(u, xj, i, hj) + dxdy_3D(u, xi, xj, j, hi, hj));
    // }

    return (mu0 / Re) * ( 2*dxj_S_ij(i, j, u) - (2.0/3.0)*dxj_S_ij(i, j, u)*kron(i, j) );
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
    // char xj;
    // xj = get_ax(j);
    // double hj;
    // hj = get_h(j);

    // int4 rho_idx;
    // rho_idx = get_sc_idx(i);

    // return (dx_3D(rho, xj, rho_idx, hj)*u[vec_buffer_idx(j)] 
    //     + rho[vec_buffer_idx(rho_idx)]*dx_3D(u, xj, j, hj));

    int4 idxs[3] = {get_sc_idx(i), j};
    global double* arrs[3] = {rho, u};

    return dx_mul_3D(2, get_ax(j), get_h(j), idxs, arrs);
}


// LES
double J_ij(int4 i, int4 j, global double* B) {
    return 0.5*(dx_3D(B, get_ax(j), i, get_h(j)) 
        - dx_3D(B, get_ax(i), j, get_h(i)));
}

double dxi_J_jk(int4 i, int4 j, int4 k, global double* B) {
    return 0.5*(dxdy_3D(B, get_ax(i), get_ax(k), j, get_h(i), get_h(k)) 
        - dxdy_3D(B, get_ax(i), get_ax(j), k, get_h(i), get_h(j)));
}

double3 dxj_j(int4 j, global double* B) {
    double result[3] = {0, 0, 0}; 
    double3 rotor;
    int4 idx[3];
    get_indxs(j, idx);

    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            for (char m = 0; m < 3; ++m) {
                result[k] += LEVI_CIVITA[k][l][m] 
                    * dxdy_3D(B, get_ax(j), char_ax[l], idx[m], get_h(j), char_h[l]);
            }
        }
    }

    rotor.x = result[0];
    rotor.y = result[1];
    rotor.z = result[2];

    return rotor;
}

double dxj_abs_j(int4 j, global double* B) {
    double3 _j = rot(j, B);
    double3 dx_j = dxj_j(j, B);
    double abs_j = sqrt(length(_j));

    return dot(_j, dx_j) / abs_j;
}

double dxj_tauB_ji(int4 i, int4 j, global double* B) {
    double result = 0;
    double3 _j = rot(j, B);
    double abs_j = sqrt(length(_j));

    result += dxj_abs_j(j, B) * J_ij(i, j, B) + dxi_J_jk(j, j, i, B)*abs_j;

    return -2*D1*SGS_DELTA_QUAD*result;
}

double abs_S_u(int4 i, global double* u){
    double result = 0;
    int4 idx[3];
    get_indxs(i, idx);

    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            double _s = S_ij(idx[k], idx[l], u);
            result += _s*_s;
        }
    }
    return sqrt(2*result);
}

double dxj_abs_S(int4 i, int4 j, global double* u) {
    double result = 0;
    int4 idx[3];
    get_indxs(i, idx);

    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            result += S_ij(idx[k], idx[l], u) * dxj_S_ij(i, j, u);
        }
    }
    return 2*result / abs_S_u(i, u);
}

double dxj_tau_u_ji(int4 i, int4 j, global double* rho, global double* u) {
    double result = 0;
    double abs_S = abs_S_u(i, u);
    result += 2*Y1*SGS_DELTA_QUAD * (
        dx_3D(rho, get_ax(j), get_sc_idx(i), get_h(j)) * abs_S * abs_S
        + 2*rho[vec_buffer_idx(get_sc_idx(i))] * abs_S * dxj_abs_S(i, j, u)
    ) * kron(i, j);

    result += -2*C1*SGS_DELTA_QUAD * (
        dx_3D(rho, get_ax(j), get_sc_idx(i), get_h(j)) * abs_S * S_ij(i, j, u)
        + rho[vec_buffer_idx(get_sc_idx(i))] * S_ij(i, j, u) * dxj_abs_S(i, j, u)
        + rho[vec_buffer_idx(get_sc_idx(i))] * abs_S * dxj_S_ij(i, j, u)
    ) * anti_kron(i, j);

    return result;
}

#endif