import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stencil_calculator import *

SHAPE = 256

GAMMA = 5.0/3.0

CFL = 0.5

dX = 1/SHAPE

shape = (SHAPE + 4,)
v_shape = (3, ) + shape

p = np.zeros(shape)
u = np.zeros(v_shape)

def v(i, u):
    return u[1,i]/u[0, i] 

def F(i, p, u):
    return np.array([u[1, i], u[1,i]*v(i, u) + p[i], (u[2, i] + p[i])*v(i, u)])

def l(i, p, u):
    return np.sqrt(GAMMA*p[i]/u[0,i]) + np.abs(v(i, u))

def l_max(p, u):
    for i in range(2, 258):
        if u[0, i] <= 0 or p[i] <= 0:
            pass # print('Alarm!')
    return max([np.sqrt(GAMMA*p[i]/u[0,i]) + np.abs(v(i, u)) for i in range(2, 258)])

def F_p(i, p, u):
    return 0.5*(F(i, p, u) + F(i+1, p, u) - max(l(i, p, u), l(i+1, p, u))*(u[:, i+1]-u[:, i]))

def ghosts(p, u):
    p[0] = p[3]
    p[1] = p[2]
    p[258] = p[257]
    p[259] = p[256]

    u[0, 0] = u[0, 3]
    u[0, 1] = u[0, 2]
    u[0, 258] = u[0, 257]
    u[0, 259] = u[0, 256]

    u[1, 0] = -u[1, 3]
    u[1, 1] = -u[1, 2]
    u[1, 258] = -u[1, 257]
    u[1, 259] = -u[1, 256]

    u[2, 0] = u[2, 3]
    u[2, 1] = u[2, 2]
    u[2, 258] = u[2, 257]
    u[2, 259] = u[2, 256]


def initial():
    u = np.zeros(v_shape)
    p = np.zeros(shape)
    for i in range(260):
        p[i] = 1.0 if i*dX < 0.5 else 0.1
        u[0, i] = 1.0 if i*dX < 0.5 else 0.125
        u[1, i] = 0.0
        u[2, i] = p[i]/(GAMMA-1) + 0.5*u[1, i]**2/u[0, i]

    ghosts(p, u)
    return p, u


def step(p, u, dT):
    new_u = np.zeros(v_shape)

    f_m = F_p(1, p, u)
    for i in range(2, 258):
        f_p = F_p(i, p, u)
        # print(new_u[:, i].shape)
        new_u[:, i] = u[:, i] - dT/dX * (f_p - f_m)
        f_m = f_p

    for i in range(2, 258):
        p[i] = new_u[2, i] - new_u[1, i]**2/new_u[0, i]

    ghosts(p, new_u)
    return p, new_u

sc_1 = Scheme(stencil=[-2, -1, 0], order=1)
sc_2 = Scheme(stencil=[-1, 0, 1], order=1)
sc_3 = Scheme(stencil=[0, 1, 2], order=1)
weno = Weno(schemes=[sc_1, sc_2, sc_3])


def weno_diff(a, x):
    # calculate betas
    betas = []
    for i, beta in enumerate(weno.beta):
        _beta = 0
        for j, b in enumerate(beta):
            _beta += b*a(x+weno.beta_scheme[i][j][0])*a(x+weno.beta_scheme[i][j][1])
        betas.append(_beta)
        
    #calculate omegas
    omegas = []
    eps = 1e-6
    for i, gamma in enumerate(weno.coefs):
        _omega = gamma / (eps+betas[i])**2
        omegas.append(_omega)
    sum_omegas = sum(omegas)
    omegas = [o/sum_omegas for o in omegas]

    #calculate diff
    diff = 0
    for i, sc in enumerate(weno.get_schemes()):
        _diff = 0
        for j, idx in enumerate(sc.get_stencil()):
            _diff += sc.coeffs[j]*a(x+idx)
        diff += omegas[i]*_diff
        # diff += weno.coefs[i] * _diff

    return diff

p = np.zeros(shape)
u = np.zeros(v_shape)

def weno_f_0(i):
    return u[1, i]

def weno_f_1(i):
    return u[1, i]*v(i, u) + p[i]

def weno_f_2(i):
    return (u[2, i] + p[i])*v(i, u)

total_steps = 0
def weno_step(p, u, dT):
    global total_steps
    # print(total_steps)
    total_steps += 1
    new_u = np.zeros(v_shape)

    for i in range(2, 258):
        new_u[:, i] = u[:, i] - dT/dX * \
            np.array([weno_diff(weno_f_0, i), weno_diff(weno_f_1, i), weno_diff(weno_f_2, i)])

    for i in range(2, 258):
        p[i] = new_u[2, i] - new_u[1, i]**2/new_u[0, i]

    ghosts(p, new_u)
    return p, new_u

X = np.linspace(0, 2*np.pi, 100)
sin_ = np.array([np.sin(x) for x in X])

def get_elem(x: int) -> float:
    if x >= 0 and x < 100:
        return sin_[x]
    elif x >= 100:
        return sin_[(x+1)%100]
    elif x < 0:
        return sin_[x-1]

def main():
    # u = np.zeros(v_shape)
    # p = np.zeros(shape)

    global p, u

    p, u = initial()

    t = 0.0

    while t < 0.2:
        dT = CFL * dX / l_max(p, u)

        p, u = weno_step(p, u, dT)
        t += dT

    x = np.linspace(0, 1, 256)
    plt.plot(x, p[2:258])
    plt.savefig(Path(f"./lab_p.jpg"))
    plt.cla()

    plt.plot(x, u[0, 2:258])
    plt.savefig(Path(f"./lab_rho.jpg"))
    plt.cla()

    plt.plot(x, u[1, 2:258]**2 / u[0, 2:258])
    plt.savefig(Path(f"./lab_v.jpg"))
    plt.cla()

    plt.plot(x, u[2, 2:258])
    plt.savefig(Path(f"./lab_e.jpg"))
    plt.cla()
    print(total_steps)
    print(f"Time: {t}")

    cos_ = []

    for i in range(100):
        cos_.append(weno_diff(a=get_elem, x=i)/(2*np.pi/100))
    plt.plot(X, sin_)
    plt.plot(X, cos_)
    plt.savefig(Path(f"./lab_test.jpg"))
    plt.cla()

main()