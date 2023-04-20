#define Re 1000
#define Rem 1000
#define Ma 10

#define gamma (7.0/5.0)

#define GHOST_CELLS 3

#define TRUE_Nx 32
#define TRUE_Ny 32
#define TRUE_Nz 32

#define L 2.0 * M_PI

#define DNS true
#define SMAGORINSKY false
#define CROSS_HELICITY false

#include "common\mhd_consts.cl"

#include ".\finite_difference\fd_flux_3d.cl"
#include ".\finite_volume\weno_fluxes.cl"
#include ".\common\initials.cl"
#include ".\common\ghost_cells.cl"
#include ".\common\integrate.cl"

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
            - c1 * dT * flux_u(index[i], u1, B1, rho1, p1)
            + c1 * dT * diff_u(index[i], u1)
        ) / RHO_2(index_x);

        B_2(index[i]) = (
            c0 * B_0(index[i]) + c1 * B_1(index[i])
            - c1 * dT * flux_B(index[i], u1, B1)
            + c1 * dT * diff_B(index[i], B1)
        );
    }
}
