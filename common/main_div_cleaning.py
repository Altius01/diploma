import os
import shutil
import sys
import pyopencl as cl
import taichi as ti

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(Path(__file__).parent.parent.as_posix())

from src.problem.new_system import System, SystemConfig


print(sys.path)

arch = ti.cpu
ti.init(arch=arch, debug=True, device_memory_GB=6)

from config import Config

PATH_CWD = Path(".")
DNS_3D_DATA_PATH = PATH_CWD / "../DNS_3D"
DNS_3D_CONFIG_PATH = PATH_CWD / "../dns_3D_config.json"

DNS_2D_DATA_PATH = PATH_CWD / "../DNS_2D"
DNS_2D_CONFIG_PATH = PATH_CWD / "../dns_2D_config.json"


def remove(path):
    """param <path> could either be relative or absolute."""
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def main2D():
    dns_2d_config = Config(file_path=DNS_2D_CONFIG_PATH)

    sys_Cfg = SystemConfig(dns_2d_config)

    if DNS_2D_DATA_PATH.exists:
        remove(DNS_2D_DATA_PATH)

    dns_2d_solver = System(sys_Cfg, data_path=DNS_2D_DATA_PATH, arch=arch)

    # dns_2d_postprocess = TiDataProcessor(
    #     config=dns_2d_config, data_path=DNS_2D_DATA_PATH
    # )

    dns_2d_solver.solve()


if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_NO_CACHE"] = "1"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

    main2D()
