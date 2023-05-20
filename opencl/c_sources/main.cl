#define Re 1000
#define Rem 100
#define Ma 1.5

#define gamma (7.0/5.0)

#define GHOST_CELLS 3

#define TRUE_Nx 32
#define TRUE_Ny 32
#define TRUE_Nz 32

#define L 2.0 * M_PI

#define DNS true
#define SMAGORINSKY false
#define CROSS_HELICITY false

#include "common/mhd_consts.cl"

#include "./finite_difference/fd_flux_3d.cl"
#include "./finite_volume/weno_fluxes.cl"
#include "./common/initials.cl"
#include "./common/ghost_cells.cl"
#include "./common/integrate.cl"

#define P_0( i )    p0[vec_buffer_idx(get_sc_idx(i))]
#define P_1( i )    p1[vec_buffer_idx(get_sc_idx(i))]
#define P_2( i )    p2[vec_buffer_idx(get_sc_idx(i))]

#define B_0( i )    B0[vec_buffer_idx(i)]
#define B_1( i )    B1[vec_buffer_idx(i)]
#define B_2( i )    B2[vec_buffer_idx(i)]

#define U_0( i )    u0[vec_buffer_idx(i)]
#define U_1( i )    u1[vec_buffer_idx(i)]
#define U_2( i )    u2[vec_buffer_idx(i)]

#define RHO_0( i )    rho0[vec_buffer_idx(get_sc_idx(i))]
#define RHO_1( i )    rho1[vec_buffer_idx(get_sc_idx(i))]
#define RHO_2( i )    rho2[vec_buffer_idx(get_sc_idx(i))]

__kernel void solver_3D_RK(
    double dT, double c0, double c1,
    double y_sgs, double c_sgs, double d_sgs,
    global double *rho0, global double *p0, 
    global double *u0, global double *B0,
    global double *rho1, global double *p1, 
    global double *u1, global double *B1,
    global double *rho2, global double *p2, 
    global double *u2, global double *B2
){
    int x = get_global_id(0); 
    int y = get_global_id(1); 
    int z = get_global_id(2);

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int4 index[3] = {index_x, index_y, index_z};

    
    RHO_2(index_x) = (
        c0 * RHO_0(index_x) + c1 * RHO_1(index_x)
        - c1 * dT * flux_rho(index_x, rho1, u1)
    );

    P_2(index_x) = pow(RHO_2(index_x), gamma);

    for (int i = 0; i < 3; ++i) {
        U_2(index[i]) = (
            c0 * U_0(index[i]) * RHO_0(index[i]) + c1 * U_1(index[i]) * RHO_1(index[i])
            - c1 * dT * flux_u(index[i], y_sgs, c_sgs, u1, B1, rho1, p1)
            + c1 * dT * diff_u(index[i], u1)
        ) / RHO_2(index_x);

        B_2(index[i]) = (
            c0 * B_0(index[i]) + c1 * B_1(index[i])
            - c1 * dT * flux_B(index[i], d_sgs, u1, B1)
            + c1 * dT * diff_B(index[i], B1)
        );
    }
}

__kernel void get_S(global double *u, global double *S) {
    int x = get_global_id(0); 
    int y = get_global_id(1); 
    int z = get_global_id(2);

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int4 index[3] = {index_x, index_y, index_z};

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            int8 idx = (int8) {i, j, x, y, z, 0, 0, 0};
            S[mat_buffer_idx(idx)] = S_ij(index[i], index[j], u);
        }
    }
}

__kernel void get_J(global double *B, global double *J) {
    int x = get_global_id(0); 
    int y = get_global_id(1); 
    int z = get_global_id(2);

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int4 index[3] = {index_x, index_y, index_z};

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            int8 idx = (int8) {i, j, x, y, z, 0, 0, 0};
            J[mat_buffer_idx(idx)] = J_ij(index[i], index[j], B);
        }
    }
}

__kernel void get_alpha(global double *rho, 
                        global double *u, 
                        global double *B, 
                        global double *alpha) 
{
    int x = get_global_id(0); 
    int y = get_global_id(1); 
    int z = get_global_id(2);

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int4 index[3] = {index_x, index_y, index_z};

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            int8 idx = (int8) {i, j, x, y, z, 0, 0, 0};

            int _sgn = (i == j) ? 1 : -1;

            #if SMAGORINSKY
            alpha[mat_buffer_idx(idx)] = _sgn * 2 * SGS_DELTA_QUAD * rho[vec_buffer_idx(index_x)] * abs_S_u(index_x, u);
            #elif CROSS_HELICITY
            alpha[mat_buffer_idx(idx)] = _sgn * 2 * SGS_DELTA_QUAD * rho[vec_buffer_idx(index_x)] * abs_f_cross(index_x, u, B);
            #endif
        }
    }
}

__kernel void get_phi(global double *u, 
                        global double *B, 
                        global double *phi) 
{
    int x = get_global_id(0); 
    int y = get_global_id(1); 
    int z = get_global_id(2);

    x += GHOST_CELLS;
    y += GHOST_CELLS;
    z += GHOST_CELLS;

    int4 index_x = (int4) {0, x, y, z};
    int4 index_y = (int4) {1, x, y, z};
    int4 index_z = (int4) {2, x, y, z};

    int4 index[3] = {index_x, index_y, index_z};

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            int8 idx = (int8) {i, j, x, y, z, 0, 0, 0};

            #if SMAGORINSKY
            phi[mat_buffer_idx(idx)] = -2 * SGS_DELTA_QUAD * abs_j(index_x, B);
            #elif CROSS_HELICITY
            double3 _j = rot(j, B);
            double3 _w = rot(j, u);

            double abs_jw = sqrt(fabs(dot(_j, _w)));

            double sgn = sign(dot(_j, _w));

            phi[mat_buffer_idx(idx)] = -2 * SGS_DELTA_QUAD * sgn * abs_jw;
            #endif
        }
    }
}
