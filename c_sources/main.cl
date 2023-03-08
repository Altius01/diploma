#include "common\mhd_consts.cl"

#include "finite_difference\fd_flux_3d.cl"
#include "finite_volume\weno_fluxes.cl"
#include "common\initials.cl"
#include "common\ghost_cells.cl"
#include "common\integrate.cl"

// #define RHO( i )    rho[vec_buffer_idx(i)]
// #define RK1( i )    rk1[vec_buffer_idx(i)]
// #define RK2( i )    rk2[vec_buffer_idx(i)]

// #define P( i )      p[vec_buffer_idx(i)]
// #define PK1( i )    pk1[vec_buffer_idx(i)]
// #define PK2( i )    pk2[vec_buffer_idx(i)]

// #define U( i )      u[vec_buffer_idx(i)]
// #define UK1( i )    uk1[vec_buffer_idx(i)]
// #define UK2( i )    uk2[vec_buffer_idx(i)]

// #define BK0( i )    B[vec_buffer_idx(i)]
// #define BK1( i )    Bk1[vec_buffer_idx(i)]
// #define BK2( i )    Bk2[vec_buffer_idx(i)]

// #define RHO_F_P( i ) rho_F_P[vec_buffer_idx(i)]
// #define U_F_P( i )   u_F_P[vec_buffer_idx(i)]
// #define B_F_P( i )   b_F_P[vec_buffer_idx(i)]

// __kernel void compute_fluxes_3D(
//     global double *rho, global double *p, 
//     global double *u, global double *B,
//     global double *rho_F_P, global double *u_F_P, global double *b_F_P
// ) {
//     int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

//     x += GHOST_CELLS - 1;
//     y += GHOST_CELLS - 1;
//     z += GHOST_CELLS - 1;

//     int4 index_x = (int4) {0, x, y, z};
//     int4 index_y = (int4) {1, x, y, z};
//     int4 index_z = (int4) {2, x, y, z};

//     int4 index[3] = {index_x, index_y, index_z};


//     RHO_F_P(index_x) = rho_F_p(index_x, rho, p, u);

//     for (int m = 0; m < 3; ++m) {
//         U_F_P(index[m]) = u_F_p(index[m], rho, p, u, B);

//         B_F_P(index[m]) = B_F_p(index[m], rho, p, u, B);
//     }
// }

// __kernel void solver_3D_RK0(
//     double dT,
//     global double *rho, global double *p, 
//     global double *u, global double *B,
//     global double *rk1, global double *pk1, 
//     global double *uk1, global double *Bk1,
//     global double *rho_F_P, global double *u_F_P, global double *b_F_P
// ) {
//     int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

//     x += GHOST_CELLS;
//     y += GHOST_CELLS;
//     z += GHOST_CELLS;

//     int4 index_x = (int4) {0, x, y, z};
//     int4 index_y = (int4) {1, x, y, z};
//     int4 index_z = (int4) {2, x, y, z};

//     int4 index[3] = {index_x, index_y, index_z};

//     rk1[vec_buffer_idx(index_x)] = (
//         RHO(index_x)
//         - dT * flux_rho(index_x, rho, u)
//         // - dT*( RHO_F_P(index_x) - RHO_F_P(shift(index_x, 0, -1)) )
//         );

//     pk1[vec_buffer_idx(index_x)] = pow(rk1[vec_buffer_idx(index_x)], gamma);

//     for (int i = 0; i < 3; ++i) {
//         uk1[vec_buffer_idx(index[i])] = ( 
//             U(index[i])*RHO(index_x)
//             // - dT*( U_F_P(index[i]) - U_F_P(shift(index[i], i, -1)) )
//             - dT * flux_u(index[i], u, B, rho, p) 
//             + dT*diff_u(index[i], u)
//         ) / rk1[vec_buffer_idx(index_x)];

//         Bk1[vec_buffer_idx(index[i])] = ( 
//             BK0(index[i]) 
//             // - dT*( B_F_P(index[i]) - B_F_P(shift(index[i], i, -1)) )
//             - dT * flux_B(index[i], u, B) 
//             + dT*diff_B(index[i], B)
//         );
//     }
// }

// __kernel void solver_3D_RK1(
//     double dT,
//     global double *rho, global double *p, 
//     global double *u, global double *B,
//     global double *rk1, global double *pk1, 
//     global double *uk1, global double *Bk1,
//     global double *rk2, global double *pk2, 
//     global double *uk2, global double *Bk2,
//     global double *rho_F_P, global double *u_F_P, global double *b_F_P
// ) {
//     int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

//     x += GHOST_CELLS;
//     y += GHOST_CELLS;
//     z += GHOST_CELLS;

//     int4 index_x = (int4) {0, x, y, z};
//     int4 index_y = (int4) {1, x, y, z};
//     int4 index_z = (int4) {2, x, y, z};

//     int4 index[3] = {index_x, index_y, index_z};

    

//     rk2[vec_buffer_idx(index_x)] = (
//         0.75*RHO(index_x) + 0.25*RK1(index_x)
//         - 0.25*dT * flux_rho(index_x, rk1, uk1)
//         // - 0.25*dT*( RHO_F_P(index_x) - RHO_F_P(shift(index_x, 0, -1)) )
//     );
//     pk2[vec_buffer_idx(index_x)] = pow(rk2[vec_buffer_idx(index_x)], gamma);

//     for (int i = 0; i < 3; ++i) {
//         uk2[vec_buffer_idx(index[i])] = (
//             0.75*U(index[i])*RHO(index_x) + 0.25*UK1(index[i])*RK1(index_x)
//             // - 0.25*dT*( U_F_P(index[i]) - U_F_P(shift(index[i], i, -1)) )
//             - 0.25*dT* flux_u(index[i], uk1, Bk1, rk1, pk1) 
//             + 0.25*dT*diff_u(index[i], uk1)
//         ) / rk2[vec_buffer_idx(index_x)];

//         Bk2[vec_buffer_idx(index[i])] = ( 0.75*BK0(index[i]) + 0.25*BK1(index[i])
//             // - 0.25*dT*( B_F_P(index[i]) - B_F_P(shift(index[i], i, -1)) )
//             - 0.25*dT* flux_B(index[i], uk1, Bk1) 
//             + 0.25*dT*diff_B(index[i], Bk1)
//         );

//     }
// }

// __kernel void solver_3D_RK2(
//     double dT,
//     global double *rho, global double *p, 
//     global double *u, global double *B,
//     global double *_rho, global double *_p, 
//     global double *_u, global double *_B,
//     global double *rk2, global double *pk2, 
//     global double *uk2, global double *Bk2,
//     global double *rho_F_P, global double *u_F_P, global double *b_F_P
// ) {
//     int x = get_global_id(0); int y = get_global_id(1); int z = get_global_id(2);

//     x += GHOST_CELLS;
//     y += GHOST_CELLS;
//     z += GHOST_CELLS;

//     int4 index_x = (int4) {0, x, y, z};
//     int4 index_y = (int4) {1, x, y, z};
//     int4 index_z = (int4) {2, x, y, z};

//     int4 index[3] = {index_x, index_y, index_z};

    

//     _rho[vec_buffer_idx(index_x)] = (
//         (1.0/3.0)*RHO(index_x) + (2.0/3.0)*RK2(index_x) 
//         - (2.0/3.0)*dT * flux_rho(index_x, rk2, uk2)
//         // - (2.0/3.0)*dT*( RHO_F_P(index_x) - RHO_F_P(shift(index_x, 0, -1)) )
//     );
//     _p[vec_buffer_idx(index_x)] = pow(_rho[vec_buffer_idx(index_x)], gamma);

//     for (int i = 0; i < 3; ++i) {
//         _u[vec_buffer_idx(index[i])] = ( 
//         (1.0/3.0)*U(index[i])*RHO(index_x) + (2.0/3.0)*UK2(index[i])*RK2(index_x) 
//         // - (2.0/3.0)*dT*( U_F_P(index[i]) - U_F_P(shift(index[i], i, -1)) )
//         - (2.0/3.0)*dT* flux_u(index[i], uk2, Bk2, rk2, pk2) 
//         + (2.0/3.0)*dT*diff_u(index[i], uk2)
//         ) / _rho[vec_buffer_idx(index_x)];

//         _B[vec_buffer_idx(index[i])] = ( (1.0/3.0)*BK0(index[i]) + (2.0/3.0)*BK2(index[i]) 
//         // - (2.0/3.0)*dT*( B_F_P(index[i]) - B_F_P(shift(index[i], i, -1)) )
//         - (2.0/3.0)*dT* flux_B(index[i], uk2, Bk2) 
//         + (2.0/3.0)*dT*diff_B(index[i], Bk2)
//         );
//     }
// }

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