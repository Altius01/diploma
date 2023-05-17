#ifndef UTILS_3D
#define UTILS_3D

void dx_3D_stencil(int4* stencil, char ax, int4 xid) {
    int4 x;
    x = xid;
    if (ax == 'x') {
        stencil[0] = (int4){x.s0, x.s1 - 2, x.s2, x.s3};
        stencil[1] = (int4){x.s0, x.s1 - 1, x.s2, x.s3};
        stencil[2] = (int4){x.s0, x.s1 + 1, x.s2, x.s3};
        stencil[3] = (int4){x.s0, x.s1 + 2, x.s2, x.s3};
    } else if (ax == 'y') {
        stencil[0] = (int4){x.s0, x.s1, x.s2 - 2, x.s3};
        stencil[1] = (int4){x.s0, x.s1, x.s2 - 1, x.s3};
        stencil[2] = (int4){x.s0, x.s1, x.s2 + 1, x.s3};
        stencil[3] = (int4){x.s0, x.s1, x.s2 + 2, x.s3};
    } else if (ax == 'z') {
        stencil[0] = (int4){x.s0, x.s1, x.s2, x.s3 - 2};
        stencil[1] = (int4){x.s0, x.s1, x.s2, x.s3 - 1};
        stencil[2] = (int4){x.s0, x.s1, x.s2, x.s3 + 1};
        stencil[3] = (int4){x.s0, x.s1, x.s2, x.s3 + 2};
    } else {
        printf("Wrong axes dx: %c!\n", ax);
    }
}

void ddx_3D_stencil(int4* stencil, char ax, int4 xid) {
    int4 x;
    x = xid;
    if (ax == 'x') {
        stencil[0] = (int4){x.s0, x.s1 - 2, x.s2, x.s3};
        stencil[1] = (int4){x.s0, x.s1 - 1, x.s2, x.s3};
        stencil[2] = (int4){x.s0, x.s1 + 0, x.s2, x.s3};
        stencil[3] = (int4){x.s0, x.s1 + 1, x.s2, x.s3};
        stencil[4] = (int4){x.s0, x.s1 + 2, x.s2, x.s3};
    } else if (ax == 'y') {
        stencil[0] = (int4){x.s0, x.s1, x.s2 - 2, x.s3};
        stencil[1] = (int4){x.s0, x.s1, x.s2 - 1, x.s3};
        stencil[2] = (int4){x.s0, x.s1, x.s2 + 0, x.s3};
        stencil[3] = (int4){x.s0, x.s1, x.s2 + 1, x.s3};
        stencil[4] = (int4){x.s0, x.s1, x.s2 + 2, x.s3};
    } else if (ax == 'z') {
        stencil[0] = (int4){x.s0, x.s1, x.s2, x.s3 - 2};
        stencil[1] = (int4){x.s0, x.s1, x.s2, x.s3 - 1};
        stencil[2] = (int4){x.s0, x.s1, x.s2, x.s3 + 0};
        stencil[3] = (int4){x.s0, x.s1, x.s2, x.s3 + 1};
        stencil[4] = (int4){x.s0, x.s1, x.s2, x.s3 + 2};
    } else {
        printf("Wrong axes ddx: %c!\n", ax);
    }
}

// void dxdy_3D_stencil(private int4* stencil, char ax, char ay, int4 xid) {
//     int4 x;
//     x = xid;
//     int idx;
//     idx = 0;
//     if ((ax == 'x' && ay == 'y') || (ax == 'y' && ay == 'x')) {
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2 - 2, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2 - 1, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2 + 1, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2 + 2, x.s3};

//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2 - 2, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2 - 1, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2 + 2, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2 + 1, x.s3};

//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2 - 2, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2 + 2, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2 - 2, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2 + 2, x.s3};

//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2 - 1, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2 + 1, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2 - 1, x.s3};
//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2 + 1, x.s3};
//     } else if ((ax == 'x' && ay == 'z') || (ax == 'z' && ay == 'x')) {
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2, x.s3 + 1};
//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2, x.s3 + 2};

//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2, x.s3 + 2};
//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2, x.s3 + 1};

//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2, x.s3 + 2};
//     stencil[idx++] = (int4){x.s0, x.s1 - 2, x.s2, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1 + 2, x.s2, x.s3 + 2};

//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2, x.s3 + 1};
//     stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2, x.s3 + 1};
//     } else if ((ax == 'y' && ay == 'z') || (ax == 'z' && ay == 'y')) {
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 1, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 2, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 2, x.s3 + 1};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 1, x.s3 + 2};

//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 1, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 2, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 1, x.s3 + 2};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 2, x.s3 + 1};

//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 2, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 2, x.s3 + 2};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 2, x.s3 - 2};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 2, x.s3 + 2};

//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 1, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 1, x.s3 + 1};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 1, x.s3 - 1};
//     stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 1, x.s3 + 1};
//     } else {
//         printf("Wrong axes dxdy: %c , %c!\n", ax, ay);
//     }
// }

void dxdy_3D_stencil(private int4* stencil, char ax, char ay, int4 xid) {
    int4 x;
    x = xid;
    
    int idx;
    idx = 0;
    if ((ax == 'x' && ay == 'y') || (ax == 'y' && ay == 'x')) {
    stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2 - 1, x.s3};
    stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2 + 1, x.s3};
    stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2 - 1, x.s3};
    stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2 + 1, x.s3};

    } else if ((ax == 'x' && ay == 'z') || (ax == 'z' && ay == 'x')) {
    stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2, x.s3 - 1};
    stencil[idx++] = (int4){x.s0, x.s1 - 1, x.s2, x.s3 + 1};
    stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2, x.s3 - 1};
    stencil[idx++] = (int4){x.s0, x.s1 + 1, x.s2, x.s3 + 1};

    } else if ((ax == 'y' && ay == 'z') || (ax == 'z' && ay == 'y')) {
    stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 1, x.s3 - 1};
    stencil[idx++] = (int4){x.s0, x.s1, x.s2 - 1, x.s3 + 1};
    stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 1, x.s3 - 1};
    stencil[idx++] = (int4){x.s0, x.s1, x.s2 + 1, x.s3 + 1};
    } else {
        printf("Wrong axes dxdy: %c , %c!\n", ax, ay);
    }
}

double dx_3D(global double* a, char ax, int4 x, double h) {
    double4 stencil;
    double4 coefs;

    int4 st_idx[4];
    dx_3D_stencil(st_idx, ax, x);

    // Записываем значения функции a в точках шаблона в массив stencil
    stencil.s0 = a[vec_buffer_idx(st_idx[0])];
    stencil.s1 = a[vec_buffer_idx(st_idx[1])];
    stencil.s2 = a[vec_buffer_idx(st_idx[2])];
    stencil.s3 = a[vec_buffer_idx(st_idx[3])];
    // Записываем коэффициенты для точек шаблона
    coefs = (double4) {
        1.0/(12.0*h), -8.0/(12.0*h),
        8.0/(12.0*h), -1.0/(12.0*h)
    };

    return dot(coefs, stencil);
}

double dx_mul_3D(int n, char ax, double h, int4* idx, global double** arrs) {
    double result = 0.0;

    for (int i = 0; i < n; ++i) {
        double sub_res = 1.0f;
        for (int j = 0; j < n; ++j) {
            sub_res *= (i == j) ? dx_3D(arrs[i], ax, idx[i], h) : arrs[j][vec_buffer_idx(idx[j])];
        }
        result += sub_res;
    }

    return result;
}

double ddx_3D(global double* a, char ax, int4 x, double h) {
    double coefs[5];

    private int4 st_idx[5];
    ddx_3D_stencil(st_idx, ax, x);

    double result;
    result = 0.0;

    // Записываем коэффициенты для точек шаблона
    coefs[0] = -1.0;
    coefs[1] = 16.0;
    coefs[2] = -30.0;
    coefs[3] = 16.0;
    coefs[4] = -1.0;
    // Считаем производную, беря точки шаблона с весами выше
    for (int i = 0; i < 5; ++i) {
        result += a[vec_buffer_idx(st_idx[i])] * coefs[i];
    }
    result /= (12.0*h*h);

    return result;
}

double dxdy_3D(global double* a, char ax, char ay, int4 x, double hx, double hy) {
    if (ax == ay)
        return ddx_3D(a, ax, x, hx); 

    double coefs[4];

    int4 st_idx[16];
    dxdy_3D_stencil(st_idx, ax, ay, x);

    double result;
    result = 0.0;

    // Записываем коэффициенты для точек шаблона
    coefs[0] = 1.0;
    coefs[1] = -1.0;
    coefs[2] = -1.0;
    coefs[3] = 1.0;

    // int idx = 0;
    // coefs[idx++] = 8.0;
    // coefs[idx++] = 8.0;
    // coefs[idx++] = 8.0;
    // coefs[idx++] = 8.0;

    // coefs[idx++] = -8.0;
    // coefs[idx++] = -8.0;
    // coefs[idx++] = -8.0;
    // coefs[idx++] = -8.0;

    // coefs[idx++] = -1.0;
    // coefs[idx++] = -1.0;
    // coefs[idx++] = 1.0;
    // coefs[idx++] = 1.0;

    // coefs[idx++] = 64.0;
    // coefs[idx++] = 64.0;
    // coefs[idx++] = -64.0;
    // coefs[idx++] = -64.0;

    // Считаем производную, беря точки шаблона с весами выше
    for (int i = 0; i < 4; ++i) {
        result += a[vec_buffer_idx(st_idx[i])] * coefs[i];
    }

    return result / (4.0*hx*hy);
}

#endif