import imageio
import vispy.plot as vp
import matplotlib.pyplot as plt
from datetime import date
from config import Config
from data_service import DataService
import numpy as np

from scipy.fft import rfft, fft, rfftfreq, fftfreq

import pyopencl as cl
import pyopencl.array as cl_array
import scipy.stats as stats

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

config = Config()

L = 2*np.pi
GHOSTS = config.GHOSTS

T_SHAPE = config.T_SHAPE

h = (L/T_SHAPE[0], L/T_SHAPE[1], L/T_SHAPE[2])

dV = h[0]*h[1]*h[2]
SHAPE = config.SHAPE
FLUX_SHAPE = (T_SHAPE[0]+1, T_SHAPE[1]+1, T_SHAPE[2]+1,)

START_STEP = config.START_STEP
steps = config.STEPS
T =  steps * 0.5*(1/T_SHAPE[0])**2
RW_DELETIMER = config.RW_DELETIMER

L = 2*np.pi
h = (L/T_SHAPE[0], L/T_SHAPE[1], L/T_SHAPE[2])
dV = h[0]*h[1]*h[2]

scalar_shape = SHAPE
vec_shape = (3,) + SHAPE

t_scalar_shape = T_SHAPE
t_vec_shape = (3,) + T_SHAPE

# 2023-03-02_LES_v1
# data_service = DataService(str(date.today()) + "_LES", scalar_shape, vec_shape)
data_service = DataService("2023-03-02_DNS", scalar_shape, vec_shape)

len_ = steps // RW_DELETIMER + 1

rho = np.zeros(scalar_shape).astype(np.float64)
u = np.zeros(vec_shape).astype(np.float64)
B_arr = np.zeros(vec_shape).astype(np.float64)
p = np.zeros(scalar_shape).astype(np.float64)


kin = np.zeros((len_,))
mag = np.zeros((len_,))

data_service.read_data(12000, (u, B_arr, rho, p))

# data_service.read_data((kin, mag))

# kin = kin
# result = rfft(kin)
# X = rfftfreq(len_, 0.5*RW_DELETIMER*(1/T_SHAPE[0])**2)

u_ = u[:, GHOSTS:-GHOSTS, GHOSTS:-GHOSTS, GHOSTS:-GHOSTS]**2

sq_u = u_[0, :]**2 + u_[1, :]**2 + u_[2, :]**2

e_kin = rho[GHOSTS:-GHOSTS, GHOSTS:-GHOSTS, GHOSTS:-GHOSTS] * sq_u

fourier_image = np.fft.fftn(e_kin)
fourier_amplitudes = np.abs(fourier_image)**2

kfreq = np.fft.fftfreq(T_SHAPE[0]) * T_SHAPE[0]
kfreq2D = np.meshgrid(kfreq, kfreq, kfreq)
knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2 + kfreq2D[2]**2)

knrm = knrm.flatten()
fourier_amplitudes = fourier_amplitudes.flatten()

kbins = np.arange(0.5, T_SHAPE[0]//2+1, 1.)
kvals = 0.5 * (kbins[1:] + kbins[:-1])
Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

if __name__ == '__main__':
    plt.loglog(kvals, Abins)
    plt.show()
    plt.cla()
    # plt.scatter(list(range(len_)), kin)
    # plt.show()

    # plotwidget.colorbar(position="top", cmap="autumn")

    # fig.show(run=True)
