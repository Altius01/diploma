import os
import pyopencl as cl
from config import Config
from solvers_new import MHD_Solver
from data_process import MHD_DataProcessor
from cl_builder import CLBuilder

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

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

DNS_256_DATA_PATH = Path.cwd() / 'DNS_256'
DNS_256_CONFIG_PATH = Path.cwd() / 'dns_256_config.json'

DNS_128_DATA_PATH = Path.cwd() / 'DNS_128'
DNS_128_CONFIG_PATH = Path.cwd() / 'dns_128_config.json'

DNS_64_DATA_PATH = Path.cwd() / 'DNS_64'
DNS_64_CONFIG_PATH = Path.cwd() / 'dns_64_config.json'

DNS_32_DATA_PATH = Path.cwd() / 'DNS_32'
DNS_32_CONFIG_PATH = Path.cwd() / 'dns_32_config.json'

def main():
    ctx = cl.create_some_context()

    dns_32_config = Config(file_path=DNS_32_CONFIG_PATH)
    dns_64_config = Config(file_path=DNS_64_CONFIG_PATH)
    dns_128_config = Config(file_path=DNS_128_CONFIG_PATH)
    dns_256_config = Config(file_path=DNS_256_CONFIG_PATH)

    # dns_256_solver = MHD_Solver(context=ctx, config=dns_256_config, 
    #                             data_path=DNS_256_DATA_PATH)
    # dns_128_solver = MHD_Solver(context=ctx, config=dns_128_config, 
    #                             data_path=DNS_128_DATA_PATH)
    # dns_64_solver = MHD_Solver(context=ctx, config=dns_64_config, 
    #                             data_path=DNS_64_DATA_PATH)
    dns_32_solver = MHD_Solver(context=ctx, config=dns_32_config, 
                                data_path=DNS_32_DATA_PATH)
    
    # dns_256_postprocess = MHD_DataProcessor(context=ctx, config=dns_256_config, 
    #                                         data_path=DNS_256_DATA_PATH)
    # dns_128_postprocess = MHD_DataProcessor(context=ctx, config=dns_128_config, 
    #                                         data_path=DNS_128_DATA_PATH)
    # dns_64_postprocess = MHD_DataProcessor(context=ctx, config=dns_64_config, 
    #                                         data_path=DNS_64_DATA_PATH)
    # dns_32_postprocess = MHD_DataProcessor(context=ctx, config=dns_32_config, 
    #                                         data_path=DNS_32_DATA_PATH)
    
    dns_32_solver.read_file(1)
    # dns_32_solver._les_filter(dns_32_solver.rho_gpu)
    # dns_32_solver._les_v_filter(dns_32_solver.B_gpu)
    dns_32_solver.get_Lu()
    # dns_32_solver.solve()
    # dns_32_postprocess.compute_energy_only()

    # dns_64_solver.solve()
    # dns_64_postprocess.compute_energy_only()

    # dns_128_solver.solve()
    # dns_128_postprocess.compute_energy_only()

    # dns_256_solver.solve()
    # dns_256_postprocess.compute_energy_only()

    # plot_energies([dns_32_postprocess, dns_64_postprocess, dns_128_postprocess, dns_256_postprocess])


def plot_energies(posprocesses: list[MHD_DataProcessor]):
    kin_e = []
    mag_e = []
    time = []
    labels = []
    for p in posprocesses:
        label = f'dns: {p.config.true_shape[0]}'
        e_k, e_m, t = p.get_energy()
        time.append(t)
        labels.append(label)
        kin_e.append(e_k)
        mag_e.append(e_m)

    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(len(kin_e)):
        plt.scatter(time[i], kin_e[i], label=labels[i])
    
    plt.gca().set(xlabel='Time', ylabel='Kinetic energy')

    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Plot of kinetic enegy by time", fontsize=22)
    plt.legend(fontsize=12)    
    plt.show()  
    plt.cla()

    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
    for i in range(len(mag_e)):
        plt.scatter(time[i], mag_e[i], label=labels[i])
    
    plt.gca().set(xlabel='Time', ylabel='Magnetic energy')

    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Plot of magnetic enegy by time", fontsize=22)
    plt.legend(fontsize=12)    
    plt.show()  


if __name__ == "__main__":
    os.environ['PYOPENCL_CTX'] = '0'
    os.environ['PYOPENCL_NO_CACHE'] = '1'
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    
    main()
