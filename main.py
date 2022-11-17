# %%
import h5py
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import pyopencl as cl

def taylor(approx, x=0):
    ret = []
    for i in range(approx):
        ret.append(np.power(x,i)/math.factorial(i))
    return ret

def taylor2(approx, x=0, y=0):
    ret = []
    for i in range(approx):
        for j in range(i+1):
            ret.append((math.comb(i, j)*np.power(x, i-j)*np.power(y, j))/math.factorial(i))
    return ret

def solve_scheme2(stencil, order, approx, a, b):
    # print(b)
    M = []

    for i in range(len(stencil)):
        if stencil[i][0] >= 0:
            s_x = '+'+str(stencil[i][0])
        else:
            s_x = str(stencil[i][0])

        if stencil[i][1] >= 0: 
            s_y = '+'+str(stencil[i][1])
        else:
            s_y = str(stencil[i][1])

        if ((a == 0 or a ==1) and (b == 0 or b ==1)):
            print(f"values.s{i} = A(x{s_x}, y{s_y}, z);")
        elif ((a == 0 or a ==2) and (b == 0 or b ==2)):
            print(f"values.s{i} = A(x{s_x}, y, z{s_y});")
        elif ((a == 2 or a ==1) and (b == 2 or b ==1)):
            print(f"values.s{i} = A(x, y{s_y}, z{s_x});")

        M.append(taylor2(approx, x=stencil[i][0], y=stencil[i][1]))

    print('\n\n')
    b = np.zeros(len(M))
    b[order] = 1
    M = np.vstack(M)

    np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

    print(M.T)
    result = np.linalg.solve(M.T, b)
    result = result.flatten()

    print("(MYFLOAT16) (", end='')
    for i in range(len(result)):
        if i % 2 == 1:
            print(str(result[i]), end=',\n')
        else:
            print(str(result[i]), end=', ')
    for i in range(15-len(result)):
        print('0.0', end=', ')
    print('0.0', end=');\n\n')

    # print(result.flatten(), '\n')
    

def solve_scheme(stencil, order, approx, a):
    M = []

    for i in range(len(stencil)):
        if stencil[i] >= 0:
            s_x = '+'+str(stencil[i])
        else:
            s_x = str(stencil[i])


        if (a == 0):
            print(f"values.s{i} = A(x{s_x}, y, z);")
        elif (a == 2):
            print(f"values.s{i} = A(x, y, z{s_x});")
        elif (a == 1):
            print(f"values.s{i} = A(x, y{s_x}, z);")

        M.append(taylor(approx, x=stencil[i]))

    print('\n\n')
    b = np.zeros(len(M))
    b[order] = 1
    print(b)
    M = np.vstack(M)

    np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

    print(M.T)
    result = np.linalg.solve(M.T, b)
    print(M.T@result)
    result = result.flatten()

    print("(MYFLOAT8) (", end='')
    for i in range(len(result)):
        if i % 2 == 1:
            print(str(result[i]), end=',\n')
        else:
            print(str(result[i]), end=', ')
    for i in range(7-len(result)):
        print('0.0', end=', ')
    print('0.0', end=');\n\n')

    print(result.flatten(), '\n')

# print(taylor(5, shift=-2))

import fractions

solve_scheme2([
    (-2, -2), (-1, -1), 
    (-1, 0), (1, 1), 
    (0, -1), (0, 1), 
    (1, -1), (1, 0), 
    (1, 1), (2, 2)], order=4, approx=4, a=0, b=2)

# solve_scheme([-2, -1, 1, 2], order=2, approx=4, a=0)
