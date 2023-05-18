import os
import sys
import pyopencl as cl
import taichi as ti

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(Path(__file__).parent.parent.as_posix())

print(sys.path)

from config import Config
from opencl.solvers_new import MHD_Solver
from data_process import MHD_DataProcessor
# from cl_builder import CLBuilder
from taichi_src.kernels.common.data_process import TiDataProcessor
from taichi_src.kernels.solver.ti_solver import TiSolver


def plot_specs():
    DNS_kin, DNS_mag = [], []
    with h5py.File("specDNS.hdf5", "r") as f:
        DNS_kin = f['kin'][:]
        DNS_mag = f['mag'][:]

    CH_kin, CH_mag = [], []
    with h5py.File("specCH.hdf5", "r") as f:
        CH_kin = f['kin'][:]
        CH_mag = f['mag'][:]

    SMAG_kin, SMAG_mag = [], []
    with h5py.File("specSMAG.hdf5", "r") as f:
        SMAG_kin = f['kin'][:]
        SMAG_mag = f['mag'][:]

    NO_kin, NO_mag = [], []
    with h5py.File("specNO.hdf5", "r") as f:
        NO_kin = f['kin'][:]
        NO_mag = f['mag'][:]

    # plt.loglog(DNS_kin[0], DNS_kin[1])
    plt.loglog(DNS_kin[0], DNS_kin[1], '-', label='DNS 180^3')
    # plt.loglog(CH_kin[0], CH_kin[1])
    plt.loglog(CH_kin[0], CH_kin[1], '-', label='CH 64^3')
    # plt.loglog(SMAG_kin[0], SMAG_kin[1])
    plt.loglog(SMAG_kin[0], SMAG_kin[1], '-', label='SMAG 64^3')
    # plt.loglog(NO_kin[0], NO_kin[1])
    plt.loglog(NO_kin[0], NO_kin[1], '-', label='LES no model 64^3')

    # plt.ylim(1e-8, 1e-1)
    # plt.xlim(9, 40)

    plt.legend()
    time = 10000 * ((np.pi/180)**2)
    plt.ylabel(f'Спектр кинетической энергии, t = {time:.4f}')
    plt.xlabel('k')

    # plt.show()
    plt.savefig('./kin_e_spectrum.jpg')
    plt.cla()

    # plt.loglog(DNS_mag[0], DNS_mag[1])
    plt.loglog(DNS_mag[0], DNS_mag[1], '-', label='DNS 180^3')
    # plt.loglog(CH_mag[0], CH_mag[1])
    plt.loglog(CH_mag[0], CH_mag[1], '-', label='CH 64^3')
    # plt.loglog(SMAG_mag[0], SMAG_mag[1])
    plt.loglog(SMAG_mag[0], SMAG_mag[1], '-', label='SMAG 64^3')
    # plt.loglog(NO_mag[0], NO_mag[1])
    plt.loglog(NO_mag[0], NO_mag[1], '-', label='LES no model 64^3')

    # plt.ylim(1e-8, 1e-1)
    # plt.xlim(9, 20)
    plt.legend()
    time = 10000 * ((np.pi/180)**2)
    plt.ylabel(f'Спектр магнитной энергии, t = {time:.4f}')
    plt.xlabel('k')

    # plt.show()
    plt.savefig('./mag_e_spectrum.jpg')
    plt.cla()


def save_spectrum(solver, idx):
    print("Start.")
    (mag_k, mag_A), (kin_k, kin_A) = solver.get_energy_spec(idx = idx)
    print("Start writing.")
    with h5py.File("spec.hdf5", "w") as f:
        dset = f.create_dataset("mag", data=(mag_k, mag_A), dtype=np.float64)
        dset = f.create_dataset("kin", data=(kin_k, kin_A), dtype=np.float64)
    pass


PATH_CWD = Path('.')
DNS_256_DATA_PATH =  PATH_CWD / 'DNS_256'
DNS_256_CONFIG_PATH = PATH_CWD / 'dns_256_config.json'

DNS_128_DATA_PATH = PATH_CWD / 'DNS_128'
DNS_128_CONFIG_PATH = PATH_CWD / 'dns_128_config.json'

DNS_64_DATA_PATH = PATH_CWD / 'DNS_64'
DNS_64_CONFIG_PATH = PATH_CWD / 'dns_64_config.json'

DNS_32_DATA_PATH = PATH_CWD / 'DNS_ti_32'
DNS_32_CONFIG_PATH = PATH_CWD / 'dns_32_config.json'

SMAG_32_DATA_PATH = PATH_CWD / 'SMAG_32'
SMAG_32_CONFIG_PATH = PATH_CWD / 'smag_32_config.json'

CROSS_32_DATA_PATH = PATH_CWD / 'CROSS_32'
CROSS_32_CONFIG_PATH = PATH_CWD / 'cross_32_config.json'

DNS_42_DATA_PATH = PATH_CWD / 'DNS_42'
DNS_42_CONFIG_PATH = PATH_CWD / 'dns_42_config.json'

def main():
    ctx = cl.create_some_context()

    cross_32_config = Config(file_path=CROSS_32_CONFIG_PATH)

    smag_32_config = Config(file_path=SMAG_32_CONFIG_PATH)

    dns_32_config = Config(file_path=DNS_32_CONFIG_PATH)
    dns_64_config = Config(file_path=DNS_64_CONFIG_PATH)
    dns_128_config = Config(file_path=DNS_128_CONFIG_PATH)
    dns_256_config = Config(file_path=DNS_256_CONFIG_PATH)

    dns_42_config = Config(file_path=DNS_42_CONFIG_PATH)

    # dns_256_solver = MHD_Solver(context=ctx, config=dns_256_config, 
    #                             data_path=DNS_256_DATA_PATH)
    # dns_128_solver = MHD_Solver(context=ctx, config=dns_128_config, 
    #                             data_path=DNS_128_DATA_PATH)
    # dns_64_solver = MHD_Solver(context=ctx, config=dns_64_config, 
    #                             data_path=DNS_64_DATA_PATH)
    # dns_32_solver = MHD_Solver(context=ctx, config=dns_32_config, 
    #                             data_path=DNS_32_DATA_PATH))

    dns_32_solver = TiSolver(config=dns_32_config, 
                                data_path=DNS_32_DATA_PATH, 
                                arch=ti.gpu
                            )

    # dns_42_solver = MHD_Solver(context=ctx, config=dns_42_config, 
    #                         data_path=DNS_42_DATA_PATH)

    # smag_32_solver = MHD_Solver(context=ctx, config=smag_32_config, 
                                # data_path=SMAG_32_DATA_PATH)

    # cross_32_solver = MHD_Solver(context=ctx, config=cross_32_config, 
    #                             data_path=CROSS_32_DATA_PATH)
    
    dns_256_postprocess = MHD_DataProcessor(context=ctx, config=dns_256_config, 
                                            data_path=DNS_256_DATA_PATH)
    dns_128_postprocess = MHD_DataProcessor(context=ctx, config=dns_128_config, 
                                            data_path=DNS_128_DATA_PATH)
    dns_64_postprocess = MHD_DataProcessor(context=ctx, config=dns_64_config, 
                                            data_path=DNS_64_DATA_PATH)
    dns_32_postprocess = TiDataProcessor(context=ctx, config=dns_32_config, 
                                            data_path=DNS_32_DATA_PATH)

    dns_42_postprocess = MHD_DataProcessor(context=ctx, config=dns_42_config, 
                                            data_path=DNS_42_DATA_PATH)

    smag_32_postprocess = MHD_DataProcessor(context=ctx, config=smag_32_config, 
                                            data_path=SMAG_32_DATA_PATH)

    cross_32_postprocess = MHD_DataProcessor(context=ctx, config=cross_32_config, 
                                data_path=CROSS_32_DATA_PATH)
    
    # smag_32_solver.solve()
    # smag_32_postprocess.compute_energy_only()

    # cross_32_solver.solve()
    # cross_32_postprocess.compute_energy_only()

    dns_32_solver.solve()
    dns_32_postprocess.compute_energy_only()

    # dns_42_solver.solve()
    # dns_42_postprocess.compute_energy_only()

    # dns_64_solver.solve()
    # dns_64_postprocess.compute_energy_only()

    # dns_128_solver.solve()
    # dns_128_postprocess.compute_energy_only()

    # dns_256_solver.solve()
    # dns_256_postprocess.compute_energy_only()

    # postprocesses = [
    #     dns_32_postprocess, dns_64_postprocess, 
    #     dns_128_postprocess, dns_256_postprocess, 
    #     smag_32_postprocess, cross_32_postprocess,
    #     dns_42_postprocess,
    # ]

    # plot_energies(postprocesses)

def plot_energies(posprocesses: list[MHD_DataProcessor]):
    kin_e = []
    mag_e = []
    time = []
    labels = []
    for p in posprocesses:
        label = ''
        label = f'{p.config.model}: {p.config.true_shape[0]}'

        e_k, e_m, t = p.get_energy()
        time.append(t)
        labels.append(label)
        kin_e.append(e_k)
        mag_e.append(e_m)

    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(len(kin_e)):
        plt.scatter(time[i], kin_e[i], label=labels[i])
    
    plt.gca().set(xlabel='t', ylabel='Кинетическая энергия')

    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Зависимость кинетической энергии от времени, Re=100, Rem=100, Ma=0.2", fontsize=22)
    plt.legend(fontsize=12)    
    plt.show()  
    plt.cla()

    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(len(mag_e)):
        plt.scatter(time[i], mag_e[i], label=labels[i])
    
    plt.gca().set(xlabel='t', ylabel='Магнитная энергия')

    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Зависимость магнитной энергии от времени, Re=100, Rem=100, Ma=0.2", fontsize=22)
    plt.legend(fontsize=12)
    plt.show()
    plt.cla()
    
    # import core_math.fft.pyfft as pyfft

    # plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
    # for i in range(len(kin_e)):
    #     plt.scatter(time[i], pyfft.fftn(arr=e_k[i]), label=labels[i])
    
    # plt.gca().set(xlabel='k', ylabel='Спект кинетической энергии')

    # plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    # plt.title("Спектр кинетической энергии", fontsize=22)
    # plt.legend(fontsize=12)
    # plt.show()
    # plt.cla()


if __name__ == "__main__":
    os.environ['PYOPENCL_CTX'] = '0'
    os.environ['PYOPENCL_NO_CACHE'] = '1'
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    
    main()
