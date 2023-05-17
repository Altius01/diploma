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

    int4 idx[3] = {get_sc_idx(i), i, j};
    global double* arrs[3] = {rho, u, u};

    return dx_mul_3D(3, xj, h, idx, arrs);
}

double dxj_BiBj(int4 i, int4 j, global double *B) {
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

    return result / (Ma*Ma);
}

double S_ij(int4 i, int4 j, global double* u) {
    return 0.5 * (dx_3D(u, get_ax(j), i, get_h(j)) + dx_3D(u, get_ax(i), j, get_h(i)));
}

double dxj_S_ij(int4 i, int4 j, global double* u) {
    return 0.5 * (ddx_3D(u, get_ax(j), i, get_h(j)) + dxdy_3D(u, get_ax(i), get_ax(j), j, get_h(i), get_h(j)));
}

double dxj_S_kl(int4 j, int4 k, int4 l, global double* u) {
    return 0.5 * (dxdy_3D(u, get_ax(j), get_ax(l), k, get_h(j), get_h(l)) 
        + dxdy_3D(u, get_ax(k), get_ax(j), l, get_h(k), get_h(j)));
}

double dxj_sigma_ij(int4 i, int4 j, global double* u){
    return (1 / Re) * ( 2*dxj_S_ij(i, j, u) - (2.0/3.0)*dxj_S_ij(i, j, u)*kron(i, j) );
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

double abs_j(int4 i, global double* B) {
    return length(rot(i, B));
}

double dxj_abs_j(int4 j, global double* B) {
    double3 _j = rot(j, B);
    double3 dx_j = dxj_j(j, B);
    double abs_j = length(_j);

    return dot(_j, dx_j) / abs_j;
}

double dxj_abs_jw(int4 j, global double* u, global double* B) {
    double3 _j = rot(j, B);
    double3 _w = rot(j, u);
    double3 dx_j = dxj_j(j, B);
    double3 dx_w = dxj_j(j, u);

    double abs_jw = sqrt(fabs(dot(_j, _w)));

    return ((dot(_j, dx_w) + dot(_w, dx_j))/(2*abs_jw + 1e-6) );
}


double dxj_tauB_ji(int4 i, int4 j, double D, global double* B) {
    double result = 0;
    double3 _j = rot(j, B);
    double abs_j = length(_j);

    if (fabs(abs_j) > 1e-20)
        result += dxj_abs_j(j, B) * J_ij(i, j, B) + dxi_J_jk(j, j, i, B)*abs_j;

    return -2*D*SGS_DELTA_QUAD*result;
}

double dxj_tauB_ji_cross(int4 i, int4 j, double D, global double* u, global double* B) {
    double result = 0;
    double3 _j = rot(j, B);
    double3 _w = rot(j, u);

    double abs_jw = sqrt(fabs(dot(_j, _w)));

    double sgn = sign(dot(_j, _w));

    result += dxj_abs_jw(j, u, B) * J_ij(i, j, B) + dxi_J_jk(j, j, i, B)*abs_jw;

    return -2*D*SGS_DELTA_QUAD*sgn*result;
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

    double abs = abs_S_u(i, u);

    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            result += S_ij(idx[k], idx[l], u) * dxj_S_kl(j, idx[k], idx[l], u);
        }
    }
    return ( 2*result / (abs + 1e-6) );
}

double abs_f_cross(int4 i, global double* u, global double* B){
    double result = 0;
    int4 idx[3];
    get_indxs(i, idx);

    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            result += S_ij(idx[k], idx[l], u) * S_ij(idx[k], idx[l], B);
        }
    }

    return sqrt(fabs(result));
}

double dxj_abs_f_cross(int4 j, global double* u, global double* B) {
    double result = 0;
    int4 idx[3];
    get_indxs(j, idx);

    double abs = abs_f_cross(j, u, B);

    for (char k = 0; k < 3; ++k) {
        for (char l = 0; l < 3; ++l) {
            result += S_ij(idx[k], idx[l], u) * dxj_S_kl(j, idx[k], idx[l], B) 
                + S_ij(idx[k], idx[l], B) * dxj_S_kl(j, idx[k], idx[l], u);
        }
    }

    return ( result / (2*abs + 1e-6) );
}

double dxj_tau_u_ji(int4 i, int4 j, double Y, double C, global double* rho, global double* u) {
    double result = 0;
    double abs_S = abs_S_u(i, u);

    if (fabs(abs_S) > 1e-20) { 
        result += 2*Y*SGS_DELTA_QUAD * (
            dx_3D(rho, get_ax(j), get_sc_idx(i), get_h(j)) * abs_S * abs_S
            + 2*rho[vec_buffer_idx(get_sc_idx(i))] * abs_S * dxj_abs_S(i, j, u)
        ) * kron(i, j);

        result += -2*C*SGS_DELTA_QUAD * (
            dx_3D(rho, get_ax(j), get_sc_idx(i), get_h(j)) * abs_S * S_ij(i, j, u)
            + rho[vec_buffer_idx(get_sc_idx(i))] * S_ij(i, j, u) * dxj_abs_S(i, j, u)
            + rho[vec_buffer_idx(get_sc_idx(i))] * abs_S * dxj_S_ij(i, j, u)
        ) * anti_kron(i, j);
    }

    return result;
}

double dxj_tau_u_lk_cross(int4 j, int4 l, int4 k, double Y, double C, global double* rho, global double* u, global double* B) {
    double result = 0;
    double abs_S = abs_S_u(l, u);
    double abs_f = abs_f_cross(l, u, B);
    double dxj_abs_f = dxj_abs_f_cross(j, u, B);
    double S = S_ij(l, k, u);

    double rho_ = rho[vec_buffer_idx(get_sc_idx(l))];

    result += 2*Y*SGS_DELTA_QUAD * (
        dx_3D(rho, get_ax(j), get_sc_idx(l), get_h(j)) * abs_f * abs_S
        + rho_ * abs_f * dxj_abs_S(l, j, u)
        + rho_ * abs_S * dxj_abs_f
    ) * kron(l, k) * 0.0;

    result += -2*C*SGS_DELTA_QUAD * (
        dx_3D(rho, get_ax(j), get_sc_idx(l), get_h(j)) * abs_f * S
        + rho_ * S * dxj_abs_f
        + rho_ * abs_f * dxj_S_kl(j, k, l, u)
    ) * anti_kron(l, k) * 0.0;

    // if (j.s1 == 3 && j.s2 == 3 && j.s3 == 10)
    //     printf("dxj_abs_f: %lf | abs_f: %lf | usl: %d \n", dxj_abs_f, abs_f, (fabs(abs_f) > 1e-8) ? 10 : 20);
    return result;
}

#endif