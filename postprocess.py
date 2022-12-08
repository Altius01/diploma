import os
import h5py
import time
import numpy as np
import pyopencl as cl
from pathlib import Path
import matplotlib
# matplotlib.use("Agg") # useful for a webserver case where you don't want to ever visualize the result live.
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from open_cl import scalar_shape, vec_shape, read_values, rho, u, B, work_dir, SHAPE

metadata = dict(title='Movie', artist='altius01')
writer = PillowWriter(fps=10, metadata=metadata)
# writer = FFMpegWriter(fps=15, metadata=metadata)

def plot_u(start_step, end_step, delimiter, ax_=0):
    fig, ax = plt.subplots()

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    _ = np.linspace(-1, 1, SHAPE[0])
    x, y = np.meshgrid(_, _)

    ax_str = {0: "x",
            1: "y",
            2: "z",}

    with writer.saving(fig, f"./{work_dir}/graphs/u/u_{ax_str[ax_]}_600.gif", 100):
        plt.style.use(['default'])
        print("Starting")
        for step in range(start_step, end_step, delimiter):
            print(step)

            read_values(step)

            plt.contourf(x,y,u.reshape(vec_shape)[ax_, :, :, 0], levels = 100, cmap='plasma')
            plt.colorbar(label='u')
            plt.xlabel('x')
            plt.ylabel('y') 
            plt.savefig(Path(f"./{work_dir}/graphs/u/u_{ax_str[ax_]}/u_{ax_str[ax_]}(x, y, 0)_{step}.jpg"))
            writer.grab_frame()
            plt.cla()
            plt.clf()


def plot_B(start_step, end_step, delimiter, ax_=0):
    fig, ax = plt.subplots()

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    _ = np.linspace(-1, 1, SHAPE[0])
    x, y = np.meshgrid(_, _)

    ax_str = {0: "x",
            1: "y",
            2: "z",}
    with writer.saving(fig, f"./{work_dir}/graphs/B/B_{ax_str[ax_]}_600.gif", 100):
        plt.style.use(['default'])
        print("Starting")
        for step in range(start_step, end_step, delimiter):
            print(step)

            read_values(step)

            plt.contourf(x,y,B.reshape(vec_shape)[ax_, :, :, 0], levels = 100, cmap='plasma')
            plt.colorbar(label='B')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(Path(f"./{work_dir}/graphs/B/B_{ax_str[ax_]}/B_{ax_str[ax_]}(x, y, 0)_{step}.jpg"))
            writer.grab_frame()
            plt.cla()
            plt.clf()


def plot_rho(start_step, end_step, delimiter):
    fig, ax = plt.subplots()

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    _ = np.linspace(-1, 1, SHAPE[0])
    x, y = np.meshgrid(_, _)

    with writer.saving(fig, f"./{work_dir}/graphs/rho/rho_600.gif", 100):
        plt.style.use(['default'])
        print("Starting")
        for step in range(start_step, end_step, delimiter):
            print(step)

            read_values(step)

            plt.contourf(x,y,u.reshape(vec_shape)[1, :, :, 0], levels = 100, cmap='plasma')
            plt.colorbar(label='rho')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(Path(f"./{work_dir}/graphs/rho/rho(x, y, 0)_{step}.jpg"))
            writer.grab_frame()
            plt.cla()
            plt.clf()

plot_rho(0, 600, 10)

plot_u(0, 600, 10, 0)
plot_u(0, 600, 10, 1)
plot_u(0, 600, 10, 2)

plot_B(0, 600, 10, 0)
plot_B(0, 600, 10, 1)
plot_B(0, 600, 10, 2)


     
