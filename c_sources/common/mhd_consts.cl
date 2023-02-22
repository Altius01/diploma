#ifndef MHD_CONSTS
#define MHD_CONSTS

#define GHOST_CELLS 3

#define Nx 134
#define Ny 134
#define Nz 134

#define L 2.0 * M_PI

__constant double const hx = L*1.0/(double)(Nx-2*GHOST_CELLS);
__constant double const hy = L*1.0/(double)(Ny-2*GHOST_CELLS);
__constant double const hz = L*1.0/(double)(Nz-2*GHOST_CELLS);

__constant double const h[3] = {hx, hy, hz};

#define gamma 5.0/3.0

#define B0 0.282094
#define p0 gamma
#define rho0 gamma*gamma
#define eps_p 0.01

#define Cs 1.0
#define Ca B0 / ( sqrt(1.25663706212e-06)*gamma )

#define Re 100.0
#define Rem 100.0
#define Ms 0.2

#define u0 (double) (Ms * Cs)

#define Ma  u0 / Ca

#define mu0  (rho0*u0*L)/Re

int vec_buffer_idx(int4 i) {
    int ax = i.s0;
    int x = i.s1;
    int y = i.s2;
    int z = i.s3;

    return ax*Nx*Ny*Nz + x*Ny*Nz + y*Nz + z;
}

inline double kron(int4 i, int4 j) {
    return (all(i == j)) ? 1.0 : 0.0;
} 

int4 get_sc_idx(int4 i) {
     int4 sc_idx;
    sc_idx = i;
    sc_idx.s0 = 0;
    return sc_idx;
}

char get_ax(int4 ax) {
    switch (ax.s0) {
        case 0:
            return 'x';
            break;
        case 1:
            return 'y';
            break;
        case 2:
            return 'z';
            break;
    }
    return '_';
}

double get_h(int4 ax) {
    switch (ax.s0) {
        case 0:
            return L*1.0/(double)(Nx-2*GHOST_CELLS);
            break;
        case 1:
            return L*1.0/(double)(Ny-2*GHOST_CELLS);
            break;
        case 2:
            return L*1.0/(double)(Nz-2*GHOST_CELLS);
            break;
    }
    return -1.0;
}

int4 shift(int4 i, int ax, int carry) {
    switch(ax) {
        case 0:
            i.s1 += carry;
            return i;
            break;
        case 1:
            i.s2 += carry;
            return i;
            break;
        case 2:
            i.s3 += carry;
            return i;
            break;
    }
    return i;
}

#endif