import os
import sys
import pyopencl as cl
import taichi as ti

from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

from new_src.new_system import TiSystem

sys.path.append(Path(__file__).parent.parent.as_posix())

print(sys.path)

arch = ti.cpu
ti.init(arch=arch, debug=True, device_memory_GB=6)

from config import Config
from taichi_src.data_process import TiDataProcessor
from taichi_src.ti_solver import TiSolver


PATH_CWD = Path(".")
DNS_3D_DATA_PATH = PATH_CWD / "../DNS_3D"
DNS_3D_CONFIG_PATH = PATH_CWD / "../dns_3D_config.json"

DNS_2D_DATA_PATH = PATH_CWD / "../DNS_2D"
DNS_2D_CONFIG_PATH = PATH_CWD / "../dns_2D_config.json"


def main3D():
    dns_3d_config = Config(file_path=DNS_3D_CONFIG_PATH)

    dns_3d_solver = TiSolver(
        config=dns_3d_config, data_path=DNS_3D_DATA_PATH, arch=arch
    )

    dns_3d_postprocess = TiDataProcessor(
        config=dns_3d_config, data_path=DNS_3D_DATA_PATH
    )

    dns_3d_solver.solve()


def main2D():
    dns_2d_config = Config(file_path=DNS_2D_CONFIG_PATH)

    dns_2d_solver = TiSystem(
        config=dns_2d_config, data_path=DNS_2D_DATA_PATH, arch=arch
    )

    dns_2d_postprocess = TiDataProcessor(
        config=dns_2d_config, data_path=DNS_2D_DATA_PATH
    )

    dns_2d_solver.solve()


if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = "0"
    os.environ["PYOPENCL_NO_CACHE"] = "1"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

    main2D()
