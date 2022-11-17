import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SHAPE = 256

GAMMA = 5.0/3.0

CFL = 1

dX = 1/SHAPE

shape = (SHAPE + 6,)
v_shape = (3, ) + shape

p = np.zeros(shape)
u = np.zeros(v_shape)

def v(i, u):
    return u[1,i]/u[0, i]

g_0 = 0.1
g_1 = 0.6
g_2 = 0.3

gammas = [g_0, g_1, g_2]

def b_0(i, a):
    return 13/12*(a(i-2)-2*a(i-1)+a(i))**2 + 0.25*(a(i-2)-4*a(i-1)+3*a(i))**2

def b_1(i, a):
    return 13/12*(a(i-1)-2*a(i)+a(i+1))**2 + 0.25*(a(i-1)-a(i+1))**2

def b_2(i, a):
    return 13/12*(a(i)-2*a(i+1)+a(i+2))**2 + 0.25*(3*a(i)-4*a(i+1)+a(i+2))**2

def js_betas(i, a):
    """
    @brief      calculates the smoothness indicators
    @param      f0     flux at cell center of cell i = 0
    @param      f1     flux at cell center of cell i = 1
    @param      f2     flux at cell center of cell i = 2
    @param      f3     flux at cell center of cell i = 3
    @param      f4     flux at cell center of cell i = 4
    @return     Returns smoothness indicators for all 3 stencils
    """
    f0 = a(i-2)
    f1 = a(i-1)
    f2 = a(i)
    f3 = a(i+1)
    f4 = a(i+2)

    beta_1 = (1./3. * (4.*f0**2 - 19.*f0*f1 +
                       25.*f1**2 + 11.*f0*f2 -
                       31.*f1*f2 + 10.*f2**2))
    beta_2 = (1./3. * (4.*f1**2 - 13.*f1*f2 +
                       13.*f2**2 + 5.*f1*f3 -
                       13.*f2*f3 + 4.*f3**2))
    beta_3 = (1./3. * (10.*f2**2 - 31.*f2*f3 +
                       25.*f3**2 + 11.*f2*f4 -
                       19.*f3*f4 + 4.*f4**2))

    return (beta_1, beta_2, beta_3)

def get_omega(g, b, eps=1e-6):
    return g/(eps+b)**2

def normilize_omegas(omegas):
    return np.array(omegas)/sum(omegas)

def sc_0(i, a):
    return 1/3*a(i-2) - 7/6*a(i-1) + 11/6*a(i)

def sc_1(i, a):
    return -1/6*a(i-1) + 5/6*a(i) + 1/3*a(i+1)

def sc_2(i, a):
    return 1/3*a(i) + 5/6*a(i+1) - 1/6*a(i+2)

def _solve(i, a):
    betas = js_betas(i, a)
    omegas = [
        get_omega(g_0, betas[0]), 
        get_omega(g_1, betas[1]),
        get_omega(g_2, betas[2]),
    ]
    omegas = normilize_omegas(omegas)

    return omegas[0]*sc_0(i, a) + \
        omegas[1]*sc_1(i, a) + omegas[2]*sc_2(i, a)

def F(i, p, u):
    # solve
    f_0 = _solve(i, lambda x: u[1, x])
    f_1 = _solve(i, lambda x: u[1,x]*v(x, u) + p[x])
    f_2 = _solve(i, lambda x: (u[2, x] + p[x])*v(x, u))
    return np.array([f_0, f_1, f_2])
    # return np.array([u[1, i], u[1,i]*v(i, u) + p[i], (u[2, i] + p[i])*v(i, u)])

def l(i, p, u):
    return np.sqrt(abs(GAMMA*p[i]/u[0,i])) + np.abs(v(i, u))

def l_max(p, u):
    return max([np.sqrt(GAMMA*p[i]/u[0,i]) + np.abs(v(i, u)) for i in range(2, 258)])

def alpha(i, u):
    return [1, max(abs(2*v(i, u)), abs(2*v(i+1, u))), max(abs(v(i, u)), abs(v(i+1, u)))]

def F_p(i, p, u):
    return 0.5*(F(i, p, u) + F(i+1, p, u) - 1.5*max(l(i, p, u), l(i+1, p, u))*(u[:, i+1]-u[:, i]))


def ghosts(p, u):
    p[0] = p[4]
    p[1] = p[4]
    p[2] = p[3]
    p[259] = p[258]
    p[260] = p[257]
    p[261] = p[256]

    u[0, 0] = u[0, 5]
    u[0, 1] = u[0, 4]
    u[0, 2] = u[0, 3]
    u[0, 259] = u[0, 258]
    u[0, 260] = u[0, 257]
    u[0, 261] = u[0, 256]

    u[1, 0] = -u[1, 5]
    u[1, 1] = -u[1, 4]
    u[1, 2] = -u[1, 3]
    u[1, 259] = -u[1, 258]
    u[1, 260] = -u[1, 257]
    u[1, 261] = -u[1, 256]

    u[2, 0] = u[2, 5]
    u[2, 1] = u[2, 4]
    u[2, 2] = u[2, 3]
    u[2, 259] = u[2, 258]
    u[2, 260] = u[2, 257]
    u[2, 261] = u[2, 256]


def initial():
    u = np.zeros(v_shape)
    p = np.zeros(shape)
    for i in range(262):
        p[i] = 1.0 if i*dX < 0.5 else 0.1
        u[0, i] = 1.0 if i*dX < 0.5 else 0.125
        u[1, i] = 0.0
        u[2, i] = p[i]/(GAMMA-1) + 0.5*u[1, i]**2/u[0, i]

    ghosts(p, u)
    return p, u

total_steps = 0

def step(p, u, dT):
    global total_steps
    total_steps += 1
    new_u = np.zeros(v_shape)

    f_m = F_p(2, p, u)
    for i in range(3, 259):
        f_p = F_p(i, p, u)
        # print(new_u[:, i].shape)
        new_u[:, i] = u[:, i] - dT/dX * (f_p - f_m)
        f_m = f_p

    for i in range(3, 259):
        p[i] = new_u[2, i] - new_u[1, i]**2/new_u[0, i]

    ghosts(p, new_u)
    return p, new_u

def main():
    u = np.zeros(v_shape)
    p = np.zeros(shape)

    p, u = initial()

    t = 0.0

    total_steps = 0
    while t < 0.1644:
        print(f"step: {total_steps}, time: {t}")
        dT = 0.5 * CFL * dX / l_max(p, u)

        p, u = step(p, u, dT)
        t += dT
        total_steps += 1

    x = np.linspace(0, 1, 256)
    plt.plot(x, p[3:259])
    plt.savefig(Path(f"./lab_p.jpg"))
    plt.cla()

    plt.plot(x, u[0, 3:259])
    plt.savefig(Path(f"./lab_rho.jpg"))
    plt.cla()

    plt.plot(x, u[1, 3:259]**2 / u[0, 3:259])
    plt.savefig(Path(f"./lab_v.jpg"))
    plt.cla()

    plt.plot(x, u[2, 3:259])
    plt.savefig(Path(f"./lab_e.jpg"))
    plt.cla()

# main()
main()