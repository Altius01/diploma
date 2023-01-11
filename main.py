# %%
import numpy as np
from stencil_calculator import Scheme, Scheme2, Weno, Weno2

np.set_printoptions(formatter={'all':lambda x: str(fractions.Fraction(x).limit_denominator())})

sc = Scheme(stencil=[-2, -1, 0], order=1)
# sc.print_coeffs()
# print(scheme_taylor(5, sc))

sc_2 = Scheme(stencil=[-1, 0, 1], order=1)
# sc_2.print_coeffs()
# print(scheme_taylor(5, sc_2))

sc_3 = Scheme(stencil=[0, 1, 2], order=1)

# sc_4 = Scheme(stencil=[-3, 0, 3], order=1)
# sc_3.print_coeffs()
# print(scheme_taylor(5, sc_3))

weno = Weno([sc, sc_2, sc_3])
print(weno.get_taylor())

xy_sc_0 = [
    (0, 0), (1, 0),
    (0, 1), (1, 1),
    (0, -1), (1, -1),
]

xy_sc_1 = [
    (-1, 0), (-1, 1),
    (0, 0), (0, 1),
    (1, 0), (1, 1),
]

xy_sc_2 = [
    (0, 1), (-1, -1),
    (0, 0), (-1, 0),
    (0, -1), (-1, -1),
]

xy_sc_3 = [
    (-1, 0), (-1, -1),
    (0, 0), (0, -1),
    (1, 0), (1, -1),
]

xy_scs = [
    Scheme2(stencil=xy_sc_0, order_x=1, order_y=1),
    Scheme2(stencil=xy_sc_1, order_x=1, order_y=1),
    Scheme2(stencil=xy_sc_2, order_x=1, order_y=1),
    Scheme2(stencil=xy_sc_3, order_x=1, order_y=1),
]

# weno2 = Weno2(xy_scs)
# print(weno2.get_taylor())

# %%
