#ifndef WENO_FLUXES
#define WENO_FLUXES

#define D 1.3

#include "weno_common_3d.cl"

double rho_F(int4 i, int4 j, global double *rho, global double *u) {
    double arr[5];
    int4 k, l;

    for (int m = 0; m < 5; ++m) {
        k = shift(i, j.s0, m-2);
        l = shift(j, j.s0, m-2);
        arr[m] = rho_uj(k, l, rho, u);
    }
    return weno_solve(arr);
}

double u_F(int4 i, int4 j, 
    global double *rho, global double *p,
    global double *u, global double *B    
) {
    double arr[5];
    int4 k, l;

    for (int m = 0; m < 5; ++m) {
        k = shift(i, j.s0, m-2);
        l = shift(j, j.s0, m-2);

        arr[m] = ( 
        rho_ui_uj(k, l, rho, u) + p_deltaij(k, l, p) 
        // + BiBj(k, l, B)
        // - sigma_ij(k, l, u)
        );
    }
    return weno_solve(arr);
}

double B_F(int4 i, int4 j,
    global double *u, global double *B    
) {
    double arr[5];
    int4 k, l;

    for (int m = 0; m < 5; ++m) {
        k = shift(i, j.s0, m-2);
        l = shift(j, j.s0, m-2);

        arr[m] = ( uiBj_ujBi(k, l, u, B) 
        //    - dxj_Bi(k, l, B)
        );
    }
    return weno_solve(arr);
}

double lambda_(int4 i, global double *rho, 
    global double *p, global double *u
) {
    int4 id_x = (int4) {0, i.s1, i.s2, i.s3};
    int4 id_y = (int4) {1, i.s1, i.s2, i.s3};
    int4 id_z = (int4) {2, i.s1, i.s2, i.s3};

    double abs_u = max(fabs(u[vec_buffer_idx(id_x)]), 
        max(fabs(u[vec_buffer_idx(id_y)]), fabs(u[vec_buffer_idx(id_z)])));
    return sqrt(fabs(gamma*p[vec_buffer_idx(id_x)]/rho[vec_buffer_idx(id_x)])) + abs_u;
}

double rho_F_p(int4 i, global double *rho, 
    global double *p, global double *u
) {
    double result = 0.0;

    double h[3] = {get_h('x'), get_h('y'), get_h('z')};

    for (int m = 0; m < 1; ++m) {
        int4 j = {m, i.s1, i.s2, i.s3};

        int4 k = shift(i, m, 1);
        int4 l = shift(j, m, 1);
        result += (0.5/h[m])*( rho_F(i, j, rho, u) + rho_F(k, l, rho, u) 
            - D*max(lambda_(k, rho, p, u), lambda_(i, rho, p, u))
            * (rho[vec_buffer_idx(k)] - rho[vec_buffer_idx(i)])
        );
    }

    return result;
}

double u_F_p(int4 i, 
    global double *rho, global double *p, 
    global double *u, global double *B
) {
    double result = 0.0;

    double h[3] = {get_h('x'), get_h('y'), get_h('z')};

    for (int m = 0; m < 1; ++m) {
        int4 j = {m, i.s1, i.s2, i.s3};

        int4 k = shift(i, m, 1);
        int4 l = shift(j, m, 1);
        result += (0.5/h[m])*( u_F(i, j, rho, p, u, B) + u_F(k, l, rho, p, u, B) 
            - D*max(lambda_(k, rho, p, u), lambda_(i, rho, p, u))
            * (u[vec_buffer_idx(k)] - u[vec_buffer_idx(i)])
            );
    }

    return result;
}

double B_F_p(int4 i, 
    global double *rho, global double *p, 
    global double *u, global double *B
) {
    double result = 0.0;

    double h[3] = {get_h('x'), get_h('y'), get_h('z')};

    for (int m = 0; m < 1; ++m) {
        int4 j = {m, i.s1, i.s2, i.s3};

        int4 k = shift(i, m, 1);
        int4 l = shift(j, m, 1);
        result += (0.5/h[m])*( B_F(i, j, u, B) + B_F(k, l, u, B)
            - D*max(lambda_(k, rho, p, u), lambda_(i, rho, p, u))
            * (B[vec_buffer_idx(k)] - B[vec_buffer_idx(i)])
            );
    }

    return result;
}

#endif