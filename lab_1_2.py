import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    return max([np.sqrt(GAMMA*p[i]/u[0,i]) + np.abs(v(i, u)) for i in range(2, 258)])

def p(i, u):
    return e(i, )

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
        p[i] = 1.0 if i*dX > 0.5 else 0.1
        u[0, i] = 1.0 if i*dX > 0.5 else 0.125
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

def main():
    u = np.zeros(v_shape)
    p = np.zeros(shape)

    p, u = initial()

    t = 0.0

    while t < 0.2:
        dT = CFL * dX / l_max(p, u)

        p, u = step(p, u, dT)
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

main()