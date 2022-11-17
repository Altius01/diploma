import time
import numpy as np
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

# from open_cl import scalar_shape, vec_shape, read_values, rho, u, B, work_dir, SHAPE
from open_cl import*

metadata = dict(title='Movie', artist='altius01')
writer = PillowWriter(fps=10, metadata=metadata)

def plot_u(start_step, end_step, delimiter, ax_=0):
    fig, ax = plt.subplots()

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    _ = np.linspace(-1, 1, SHAPE[0])
    x, y = np.meshgrid(_, _)

    ax_str = {0: "x",
            1: "y",
            2: "z",}

    with writer.saving(fig, f"./{work_dir}/graphs/u/u_{ax_str[ax_]}_.gif", 100):
        plt.style.use(['default'])
        print("Starting")
        for step in range(start_step, end_step, delimiter):
            print(step)

            global u, B, rho
            data_service = DataService(date.today(), scalar_shape, vec_shape)
            data_service.read_data(i, (u, B, rho))

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
    with writer.saving(fig, f"./{work_dir}/graphs/B/B_{ax_str[ax_]}.gif", 100):
        plt.style.use(['default'])
        print("Starting")
        for step in range(start_step, end_step, delimiter):
            print(step)

            global u, B, rho
            data_service = DataService(date.today(), scalar_shape, vec_shape)
            data_service.read_data(i, (u, B, rho))

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

    with writer.saving(fig, f"./{work_dir}/graphs/rho/rho.gif", 100):
        plt.style.use(['default'])
        print("Starting")
        for step in range(start_step, end_step, delimiter):
            print(step)

            global u, B, rho
            data_service = DataService(date.today(), scalar_shape, vec_shape)
            data_service.read_data(i, (u, B, rho))

            plt.contourf(x,y,rho.reshape(scalar_shape)[:, :, 0], levels = 100, cmap='plasma')
            plt.colorbar(label='rho')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(Path(f"./{work_dir}/graphs/rho/rho(x, y, 0)_{step}.jpg"))
            writer.grab_frame()
            plt.cla()
            plt.clf()

#  test()

start_step = 0
# main(start_step)

for i in range(start_step, start_step + STEPS, 100):
    # print(i, i+100)
    compute_kinetic_energy(i, i+100)
    # sum_ro(i, i+100)

plot_rho(start_step, STEPS, 10)

plot_u(start_step, STEPS, 10, 0)
plot_u(start_step, STEPS, 10, 1)
plot_u(start_step, STEPS, 10, 2)

plot_B(start_step, STEPS, 10, 0)
plot_B(start_step, STEPS, 10, 1)
plot_B(start_step, STEPS, 10, 2)


     
