#ifndef MHD_CONSTS
#define MHD_CONSTS

#define Nx (TRUE_Nx + 2*GHOST_CELLS)
#define Ny (TRUE_Ny + 2*GHOST_CELLS)
#define Nz (TRUE_Nz + 2*GHOST_CELLS)

__constant double const hx = L*1.0/(double)(TRUE_Nx);
__constant double const hy = L*1.0/(double)(TRUE_Ny);
__constant double const hz = L*1.0/(double)(TRUE_Nz);

__constant double const h[3] = {hx, hy, hz};

__constant double const dV = hx*hy*hz;

#define eps_p 0.01

// #define C1  0.17
// #define Y1  0.1
// #define D1  0.1

// #define C3  0.173
// #define Y3  0.1
// #define D3  0.1

#define SGS_DELTA_QUAD   (hx*hx + hy*hy + hz*hz)
#define SGS_DELTA_ABS   sqrt(hx*hx + hy*hy + hz*hz)


int t_vec_buffer_idx(int4 i) {
    int ax = i.s0;
    int x = i.s1;
    int y = i.s2;
    int z = i.s3;

    return ax*TRUE_Nx*TRUE_Ny*TRUE_Nz + x*TRUE_Ny*TRUE_Nz + y*TRUE_Nz + z;
}

int t_mat_buffer_idx(int8 idx) {
    int i = idx.s0;
    int j = idx.s1;
    int x = idx.s2;
    int y = idx.s3;
    int z = idx.s4;

    return (i*3 + j)*TRUE_Nx*TRUE_Ny*TRUE_Nz + x*TRUE_Ny*TRUE_Nz + y*TRUE_Nz + z;
}


int vec_buffer_idx(int4 i) {
    int ax = i.s0;
    int x = i.s1;
    int y = i.s2;
    int z = i.s3;

    return ax*Nx*Ny*Nz + x*Ny*Nz + y*Nz + z;
}

int mat_buffer_idx(int8 idx) {
    int i = idx.s0;
    int j = idx.s1;
    int x = idx.s2;
    int y = idx.s3;
    int z = idx.s4;

    return (i*3 + j)*Nx*Ny*Nz + x*Ny*Nz + y*Nz + z;
}

inline double kron(int4 i, int4 j) {
    return (all(i == j)) ? 1.0 : 0.0;
} 

inline double anti_kron(int4 i, int4 j) {
    return (all(i == j)) ? 0.0 : 1.0;
} 

int4 change_idx_axe(int4 i, char axes) {
    int4 sc_idx;
    sc_idx = i;
    sc_idx.s0 = axes;
    return sc_idx;
}

void get_indxs(int4 i, int4* idxs) {
    idxs[0] = change_idx_axe(i, 0);
    idxs[1] = change_idx_axe(i, 1);
    idxs[2] = change_idx_axe(i, 2); 
} 

int4 get_sc_idx(int4 i) {
    int4 sc_idx;
    sc_idx = i;
    sc_idx.s0 = 0;
    return sc_idx;
}

__constant char char_ax[3] = {'x', 'y', 'z'};

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

__constant double char_h[3] = {L*1.0/(double)(Nx-2*GHOST_CELLS), 
    L*1.0/(double)(Ny-2*GHOST_CELLS), 
    L*1.0/(double)(Nz-2*GHOST_CELLS)};

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

__constant double LEVI_CIVITA[3][3][3] = {
{
    {0, 0, 0},
    {0, 0, 1},
    {0, -1, 0}
},
{
    {0, 0, -1},
    {0, 0, 0},
    {1, 0, 0}
},
{
    {0, 1, 0},
    {-1, 0, 0},
    {0, 0, 0}
}
};

double levi_civita(int4 i, int4 j, int4 k) {
    return LEVI_CIVITA[i.s0][j.s0][k.s0];
}

#endif