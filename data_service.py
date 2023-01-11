import h5py
import numpy as np
from pathlib import Path

class DataService:
    def __init__(self, dir_name, scalar_shape, vec_shape) -> None:
        self.dir_name = f"./{dir_name}"
        self.scalar_shape = scalar_shape
        self.vector_shape = vec_shape
        self._create_dirs()

    def _create_dirs(self):
        if not Path(f"{self.dir_name}/data/").exists():
            Path(f"{self.dir_name}/data/").mkdir(parents=True, exist_ok=True)
        if not Path(f"{self.dir_name}/graphs/").exists():
            Path(f"{self.dir_name}/graphs/").mkdir(parents=True, exist_ok=True)

        if not Path(f"{self.dir_name}/graphs/B").exists():
            Path(f"{self.dir_name}/graphs/B").mkdir(parents=True, exist_ok=True)
        if not Path(f"{self.dir_name}/graphs/u").exists():
            Path(f"{self.dir_name}/graphs/u").mkdir(parents=True, exist_ok=True)
        if not Path(f"{self.dir_name}/graphs/rho").exists():
            Path(f"{self.dir_name}/graphs/rho").mkdir(parents=True, exist_ok=True)

    def set_folder_name(self, dir_name):
        self.dir_name = f"./{dir_name}"

    def save_data(self, step, args):
        with h5py.File(Path(f"{self.dir_name}/data/step_{step}.hdf5"), "w") as f:
            dset = f.create_dataset("u", self.vector_shape, dtype=np.float32)
            dset[:] = args[0].reshape(self.vector_shape)
            dset = f.create_dataset("B", self.vector_shape, dtype=np.float32)
            dset[:] = args[1].reshape(self.vector_shape)
            dset = f.create_dataset("rho", self.scalar_shape, dtype=np.float32)
            dset[:] = args[2].reshape(self.scalar_shape)

    def read_data(self, step, args):
        with h5py.File(Path(f"{self.dir_name}/data/step_{step}.hdf5"), "r") as f:
            args[0].reshape(self.vector_shape)[:] = f['u']
            args[1].reshape(self.vector_shape)[:] = f['B']
            args[2].reshape(self.scalar_shape)[:] = f['rho']
