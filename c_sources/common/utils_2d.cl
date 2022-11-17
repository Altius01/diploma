#ifndef UTILS_1D
#define UTILS_1D

void dx_2D_stencil( int3* stencil, char ax, int3 xid) {
     int3 x;
    x = xid;
    if (ax == 'x') {
        stencil[0] = (int3){x.s0, x.s1 - 2, x.s2};
        stencil[1] = (int3){x.s0, x.s1 - 1, x.s2};
        stencil[2] = (int3){x.s0, x.s1 + 1, x.s2};
        stencil[3] = (int3){x.s0, x.s1 + 2, x.s2};
    } else if (ax == 'y') {
        stencil[0] = (int3){x.s0, x.s1, x.s2 - 2};
        stencil[1] = (int3){x.s0, x.s1, x.s2 - 1};
        stencil[2] = (int3){x.s0, x.s1, x.s2 + 1};
        stencil[3] = (int3){x.s0, x.s1, x.s2 + 2};
    } else {
        printf("Wrong axes!\n");
    }
}

void ddx_2D_stencil( int3* stencil, char ax, int3 xid) {
     int3 x;
    x = xid;
    if (ax == 'x') {
        stencil[0] = (int3){x.s0, x.s1 - 2, x.s2};
        stencil[1] = (int3){x.s0, x.s1 - 1, x.s2};
        stencil[2] = (int3){x.s0, x.s1, x.s2};
        stencil[3] = (int3){x.s0, x.s1 + 1, x.s2};
        stencil[3] = (int3){x.s0, x.s1 + 2, x.s2};
    } else if (ax == 'y') {
        stencil[0] = (int3){x.s0, x.s1, x.s2 - 2};
        stencil[1] = (int3){x.s0, x.s1, x.s2 - 1};
        stencil[2] = (int3){x.s0, x.s1, x.s2};
        stencil[3] = (int3){x.s0, x.s1, x.s2 + 1};
        stencil[3] = (int3){x.s0, x.s1, x.s2 + 2};
    } else {
        printf("Wrong axes!\n");
    }
}

void dxdy_2D_stencil( int3* stencil, int3 xid) {
     int3 x;
    x = xid;
     int idx;
    idx = 0;
    stencil[idx++] = (int3){x.s0, x.s1 + 1, x.s2 - 2};
    stencil[idx++] = (int3){x.s0, x.s1 + 2, x.s2 - 1};
    stencil[idx++] = (int3){x.s0, x.s1 - 2, x.s2 + 1};
    stencil[idx++] = (int3){x.s0, x.s1 - 1, x.s2 + 2};

    stencil[idx++] = (int3){x.s0, x.s1 - 1, x.s2 - 2};
    stencil[idx++] = (int3){x.s0, x.s1 - 2, x.s2 - 1};
    stencil[idx++] = (int3){x.s0, x.s1 + 1, x.s2 + 2};
    stencil[idx++] = (int3){x.s0, x.s1 + 2, x.s2 + 1};

    stencil[idx++] = (int3){x.s0, x.s1 + 2, x.s2 - 2};
    stencil[idx++] = (int3){x.s0, x.s1 - 2, x.s2 + 2};
    stencil[idx++] = (int3){x.s0, x.s1 - 2, x.s2 - 2};
    stencil[idx++] = (int3){x.s0, x.s1 + 2, x.s2 + 2};

    stencil[idx++] = (int3){x.s0, x.s1 - 1, x.s2 - 1};
    stencil[idx++] = (int3){x.s0, x.s1 + 1, x.s2 + 1};
    stencil[idx++] = (int3){x.s0, x.s1 + 1, x.s2 - 1};
    stencil[idx++] = (int3){x.s0, x.s1 - 1, x.s2 + 1};
}

double dx_2D(arr3 a, char ax, int3 x, double h) {
     double4 stencil;
     double4 coefs;

     int3 st_idx[4];
    dx_2D_stencil(st_idx, ax, x);

    // Записываем значения функции a в точках шаблона в массив stencil
    stencil.s0 = a(st_idx[0]);
    stencil.s1 = a(st_idx[1]);
    stencil.s2 = a(st_idx[2]);
    stencil.s3 = a(st_idx[3]);
    // Записываем коэффициенты для точек шаблона
    coefs = (double4) {
        1.0/(12.0*h), 8.0/(12.0*h),
        8.0/(12.0*h), -1.0/(12.0*h)
    };

    return dot(stencil, coefs);
}

double ddx_2D(arr3 a, char ax, int3 x, double h) {
     double coefs[5];

     int3 st_idx[5];
    dx_2D_stencil(st_idx, ax, x);

     double result;
    result = 0.0;

    // Записываем коэффициенты для точек шаблона
    coefs[0] = -1.0/(12.0*h*h);
    coefs[1] = 16.0/(12.0*h*h);
    coefs[2] = 30.0/(12.0*h*h);
    coefs[3] = 16.0/(12.0*h*h);
    coefs[4] = -1.0/(12.0*h*h);
    // Считаем производную, беря точки шаблона с весами выше
    for (int i = 0; i < 5; ++i) {
        result += a(st_idx[i]) * coefs[i];
    }

    return result;
}

double dxdy_2D(arr3 a, int3 x, double hx, double hy) {
     double coefs[5];

     int3 st_idx[5];
    dxdy_2D_stencil(st_idx, x);

     double result;
    result = 0.0;

    // Записываем коэффициенты для точек шаблона
    coefs[0] = 8.0/(144.0*hx*hy);
    coefs[1] = -8.0/(144.0*hx*hy);
    coefs[2] = -1.0/(144.0*hx*hy);
    coefs[3] = 64.0/(144.0*hx*hy);
    // Считаем производную, беря точки шаблона с весами выше
    for (int i = 0; i < 16; ++i) {
        result += a(st_idx[i]) * coefs[(int)i/4];
    }

    return result;
}

#endif