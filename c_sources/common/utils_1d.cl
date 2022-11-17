#ifndef UTILS_1D
#define UTILS_1D

void dx_1D_stencil( int2* stencil, char ax, int2 xid) {
     int2 x;
    x = xid;
    if (ax == 'x') {
        stencil[0] = (int2){x.s0, x.s1 - 2};
        stencil[1] = (int2){x.s0, x.s1 - 1};
        stencil[2] = (int2){x.s0, x.s1 + 1};
        stencil[3] = (int2){x.s0, x.s1 + 2};
    } else {
        printf("Wrong axes!\n");
    }
}

void ddx_1D_stencil( int2* stencil, char ax, int2 xid) {
     int2 x;
    x = xid;
    if (ax == 'x') {
        stencil[0] = (int2){x.s0, x.s1 - 2};
        stencil[1] = (int2){x.s0, x.s1 - 1};
        stencil[2] = (int2){x.s0, x.s1};
        stencil[3] = (int2){x.s0, x.s1 + 1};
        stencil[3] = (int2){x.s0, x.s1 + 2};
    } else {
        printf("Wrong axes!\n");
    }
}

double dx_1D(arr2 a, char ax, int2 x, double h) {
     double4 stencil;
     double4 coefs;

     int2 st_idx[4];
    dx_1D_stencil(st_idx, ax, x);

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

double ddx_1D(arr2 a, char ax, int2 x, double h) {
     double coefs[5];

     int2 st_idx[5];
    dx_1D_stencil(st_idx, ax, x);

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

#endif