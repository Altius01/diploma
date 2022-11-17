#include "my_math.h"

inline int get_axis(int gid)
{
    return (gid /  (SIZE_X*SIZE_Y*SIZE_Z));
}

inline int get_z(int gid)
{
    return  (gid %  (SIZE_X*SIZE_Y*SIZE_Z)) / (SIZE_X*SIZE_Y);
}

inline int get_y(int gid)
{
    return ((gid %  (SIZE_X*SIZE_Y*SIZE_Z)) %  (SIZE_X*SIZE_Y)) / SIZE_X;
}

inline int get_x(int gid)
{
    return ((gid %  (SIZE_X*SIZE_Y*SIZE_Z)) %  (SIZE_X*SIZE_Y)) % SIZE_X;
}

inline int index(int gid, char axis, char component, int shift)
{
    int carry = shift;
    switch (component)
    {
        case 1:
            carry *= SIZE_X;
            break;
        case 2:
            carry *= SIZE_X*SIZE_Y;
            break;
    }

    carry += axis*SIZE_X*SIZE_Y*SIZE_Z;
    return gid + carry;
}

MYFLOAT d(__global MYFLOAT *a, MYFLOAT* coefs, int* idx)
{
    MYFLOAT res = 0;
    for (int i = 0; i < 5; ++i)
    {
        res += coefs[i]*a[idx[i]];
    }
    return res/DELTA_H;
}

MYFLOAT central_d(__global MYFLOAT *a, int gid, char axis, char component)
{
    int idx[] = {
        index(gid, axis, component, -1), 
        index(gid, axis, component, -2), 
        gid,
        index(gid, axis, component, 1), 
        index(gid, axis, component, 2)
    };
    MYFLOAT coefs[] = {1/28, -4/7, 0, 4/7, -1/28};
    return d(a, coefs, idx);
}

MYFLOAT left_d(__global MYFLOAT *a, int gid, char axis, char component)
{
    int idx[] = {
        gid, 
        index(gid, axis, component, -1), 
        index(gid, axis, component, -2), 
        index(gid, axis, component, -3), 
        index(gid, axis, component, -4)
    };
    MYFLOAT coefs[] = {25/12, -4, 3, -4/3, 1/4};
    return d(a, coefs, idx);
}

MYFLOAT right_d(__global MYFLOAT *a, int gid, char axis, char component)
{
    int idx[] = {
        gid, 
        index(gid, axis, component, 1), 
        index(gid, axis, component, 2), 
        index(gid, axis, component, 3), 
        index(gid, axis, component, 4)
    };
    MYFLOAT coefs[] = {-25/12, 4, -3, 4/3, -1/4};
    return d(a, coefs, idx);
}

MYFLOAT d_x(__global MYFLOAT *a, int gid, char axis)
{
    if (get_x(gid) >= SIZE_X-2) {
        return left_d(a, gid, axis, 0);
    } else if (get_x(gid) <= 1) {
        return right_d(a, gid, axis, 0); 
    } else {
        return central_d(a, gid, axis, 0);
    }
}

MYFLOAT d_y(__global MYFLOAT *a, int gid, char axis)
{
    if (get_y(gid) >= SIZE_Y-2) {
        return left_d(a, gid, axis, 1);
    } else if (get_y(gid) <= 1) {
        return right_d(a, gid, axis, 1); 
    } else {
        return central_d(a, gid, axis, 1);
    }
}

MYFLOAT d_z(__global MYFLOAT *a, int gid, char axis)
{
    if (get_z(gid) >= SIZE_Z-2) {
        return left_d(a, gid, axis, 2);
    } else if (get_z(gid) <= 1) {
        return right_d(a, gid, axis, 2); 
    } else {
        return central_d(a, gid, axis, 2);
    }
}

MYFLOAT d_i(__global MYFLOAT *a, char i, int gid, char axis)
{
    switch (i) {
        case 0:
            return d_x(a, gid, axis);
            break;
        case 1:
            return d_y(a, gid, axis);
            break;
        default:
            return d_z(a, gid, axis);
    }
}

MYFLOAT div(__global MYFLOAT* a, int gid, char axis)
{
   return d_x(a, gid, axis) + d_y(a, gid, axis) + d_z(a, gid, axis); 
}