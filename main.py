# %%
import math
import numpy as np
import sympy
import matplotlib.pyplot as plt

def taylor(approx, x=0):
    ret = []
    for i in range(approx):
        ret.append(np.power(x,i)/math.factorial(i))
    return np.array(ret)


def taylor2(approx, x=0, y=0):
    ret = []
    for i in range(approx):
        for j in range(i+1):
            ret.append((math.comb(i, j)*np.power(x, i-j)*np.power(y, j))/math.factorial(i))
    return ret


def solve_scheme2(stencil, order, a=0, b=1):
    M = []

    approx = order

    for i in range(len(stencil)):
        M.append(taylor2(approx, x=stencil[i][0], y=stencil[i][1]))

    b = np.zeros(len(M))
    b[order] = 1
    M = np.vstack(M)

    np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

    # print(M.T)
    result = np.linalg.solve(M.T, b)
    return np.array(result.flatten())


def solve_scheme(stencil, order, a=0):
    M = []

    approx = len(stencil)

    for i in range(len(stencil)):
        M.append(taylor(approx, x=stencil[i]))

    b = np.zeros(len(M))
    b[order] = 1
    M = np.vstack(M)

    # print(M.T)
    result = np.linalg.solve(M.T, b)
    # print(M.T@result)
    # print(np.array(result.flatten()))
    return np.array(result.flatten())

# print(taylor(5, shift=-2))

import fractions

# solve_scheme2([
#     (-2, -2), (-1, -1), 
#     (-1, 0), (1, 1), 
#     (0, -1), (0, 1), 
#     (1, -1), (1, 0), 
#     (1, 1), (2, 2)], order=4, approx=4, a=0, b=2)

# scheme = solve_scheme([-2, -1, 1, 2], order=2, a=0)

# print(scheme)
# print(scheme_taylor(4, [scheme], [2]))


class Scheme:
    order_x = None
    coeffs = None
    _stencil = None
    approx = None

    def __init__(self, stencil, order, coeffs=None):
        self._stencil = stencil
        self.approx = len(stencil)
        self.order_x = order

        if coeffs:
            self.coeffs = coeffs
        else:
            self._calculate_coeffs()


    def _calculate_coeffs(self):
        self.coeffs = solve_scheme(order=self.order_x, stencil=self._stencil)


    def set_stencil(self, stencil):
        self._stencil = stencil
        self._calculate_coeffs()


    def get_stencil(self):
        return self._stencil


    def taylor(self, n):
        if self.coefs and self._stencil:
            result = self.coeffs[0]*taylor(n, self._stencil[0])
            for i, s in enumerate(self._stencil[1:]):
                result += self.coeffs[i]*np.array(taylor(n, s))
            return result


    def print_stencil(self, ax):
        np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

        for i in range(len(self._stencil)):
            if self._stencil[i] >= 0:
                s_x = '+'+str(self._stencil[i])
            else:
                s_x = str(self._stencil[i])

            if (ax == 0):
                print(f"values.s{i} = A(x{s_x}, y, z);")
            elif (ax == 2):
                print(f"values.s{i} = A(x, y, z{s_x});")
            elif (ax == 1):
                print(f"values.s{i} = A(x, y{s_x}, z);")


    def print_coeffs(self):
        np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
        print(f"Coefs: {self.coeffs}\n")
        print("(MYFLOAT8) (", end='')
        for i in range(len(self.coeffs)):
            if i % 2 == 1:
                print(str(self.coeffs[i]), end=',\n')
            else:
                print(str(self.coeffs[i]), end=', ')
        for i in range(7-len(self.coeffs)):
            print('0.0', end=', ')
        print('0.0', end=');\n\n')


class Scheme2(Scheme):
    order_y = None

    def __init__(self, stencil, order_x, order_y, coeffs=None):
        self.order_y = order_y
        Scheme.__init__(self, stencil, order_x, coeffs)

    def _calculate_coeffs(self):
        n = self.order_x+self.order_y
        order = int((n/2)*(2*1 + (n-1)) + self.order_y)
        self.coeffs = solve_scheme2(order=order, stencil=self._stencil)

    def taylor(self, n):
        if self.coefs and self._stencil:
            result = self.coeffs[0]*taylor2(n, self._stencil[0][0], self._stencil[0][1])
            for i, s in enumerate(self._stencil[1:]):
                result += self.coeffs[i]*np.array(taylor2(n, s[0], s[1]))
            return result

    def print_stencil(self, ax_0, ax_1):
        np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

        for i in range(len(self._stencil)):
            if self._stencil[i][0] >= 0:
                s_x = '+'+str(self._stencil[i][0])
            else:
                s_x = str(self._stencil[i][0])

            if self._stencil[i][1] >= 0: 
                s_y = '+'+str(self._stencil[i][1])
            else:
                s_y = str(self._stencil[i][1])

            if ((ax_0 == 0 or ax_0 ==1) and (ax_1 == 0 or ax_1 ==1)):
                print(f"values.s{i} = A(x{s_x}, y{s_y}, z);")
            elif ((ax_0 == 0 or ax_0 ==2) and (ax_1 == 0 or ax_1 ==2)):
                print(f"values.s{i} = A(x{s_x}, y, z{s_y});")
            elif ((ax_0 == 2 or ax_0 ==1) and (ax_1 == 2 or ax_1 ==1)):
                print(f"values.s{i} = A(x, y{s_y}, z{s_x});")

    def print_coeffs(self):
        np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
        print(f"Coefs: {self.coeffs}\n")
        print("(MYFLOAT16) (", end='')
        for i in range(len(self.coeffs)):
            if i % 2 == 1:
                print(str(self.coeffs[i]), end=',\n')
            else:
                print(str(self.coeffs[i]), end=', ')
        for i in range(15-len(self.coeffs)):
            print('0.0', end=', ')
        print('0.0', end=');\n\n')


stencil2 = [
    (-2, -2), (-1, -1), 
    (-1, 0), (1, 1), 
    (0, -1), (0, 1), 
    (1, -1), (1, 0), 
    (1, 1), (2, 2)
]

# sc2 = Scheme2(stencil=stencil2, order_x=1, order_y=1)
# sc2.print_coeffs()

def scheme_taylor(approx, scheme):
    result = np.zeros(approx)
    for i, point in enumerate(scheme.get_stencil()):
        result += scheme.coeffs[i]*taylor(approx, point)
    return result
    

def scheme2_taylor(approx, scheme):
    result = np.zeros(approx)
    for i, point in enumerate(scheme.get_stencil()):
        result += scheme.coeffs[i]*taylor2(approx, point[0], point[1])
    return result

np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})
sc = Scheme(stencil=[-2, -1, 0], order=1)
# sc.print_coeffs()
# print(scheme_taylor(5, sc))

sc_2 = Scheme(stencil=[-1, 0, 1], order=1)
# sc_2.print_coeffs()
# print(scheme_taylor(5, sc_2))

sc_3 = Scheme(stencil=[0, 1, 2], order=1)

sc_4 = Scheme(stencil=[-3, 0, 3], order=1)
# sc_3.print_coeffs()
# print(scheme_taylor(5, sc_3))

def solve_weno(schemes: list[Scheme], order):
    M = []

    approx = np.min([s.approx for s in schemes]) + len(schemes) - 1
    for sc in schemes:
        M.append(scheme_taylor(approx, sc))
    M = np.vstack(M)
    M = M.T
    M = M[~np.all(M == 0, axis=1)]

    b = np.zeros(len(M))
    b[0] = 1

    result = np.linalg.solve(M, b)
    return np.array(result.flatten())


def weno_taylor(coeffs, schemes):
    approx = np.min([s.approx for s in schemes]) + len(schemes) - 1
    result = np.zeros(approx)
    for i, s in enumerate(schemes):
        result += coeffs[i]*scheme_taylor(approx, s)
    return result

coeffs_weno = solve_weno([sc, sc_2, sc_3, sc_4], 1)
print(coeffs_weno)

print(weno_taylor(coeffs_weno, [sc, sc_2, sc_3, sc_4]))