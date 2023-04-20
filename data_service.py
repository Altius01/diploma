import h5py
import numpy as np
from pathlib import Path

class DataService:
    rw_energy = True
    _dirs_to_create = ['data/', 'graphs/rho', 'graphs/B', 'graphs/u', 'energy/',]

    def __init__(self, scalar_shape=None, vec_shape=None, dir_name='', rw_energy=True) -> None:
        self.dir_name = f"./{dir_name}"
        self.scalar_shape = scalar_shape
        self.vector_shape = vec_shape
        self.rw_energy = rw_energy
        self._create_dirs()

    def _create_dirs(self):
        for dir in self._dirs_to_create:
            if not (Path(f'{self.dir_name}') / dir).exists():
                (Path(f'{self.dir_name}') / dir).mkdir(parents=True, exist_ok=True)
            

    def set_folder_name(self, dir_name):
        self.dir_name = f"./{dir_name}"

    def save_data(self, step, args):
        with h5py.File(Path(f"{self.dir_name}/data/step_{step}.hdf5"), "w") as f:
            dset = f.create_dataset("t", data=np.array([args[0]]), dtype=np.float64)
            dset = f.create_dataset("rho", data=args[1].get(), dtype=np.float64)
            dset = f.create_dataset("p", data=args[2].get(), dtype=np.float64)
            dset = f.create_dataset("u", data=args[3].get(), dtype=np.float64)
            dset = f.create_dataset("B", data=args[4].get(), dtype=np.float64)

    def read_data(self, step): # args):
        with h5py.File(Path(f"{self.dir_name}/data/step_{step}.hdf5"), "r") as f:
            # args[0].reshape(self.vector_shape)[:] = f['u']
            # args[1].reshape(self.vector_shape)[:] = f['B']
            # args[2].reshape(self.scalar_shape)[:] = f['rho']
            # args[3].reshape(self.scalar_shape)[:] = f['p']
            # args[0][0].set(ary=f['rho'][:], queue=args[0][1])
            # args[1][0].set(ary=f['p'][:], queue=args[1][1])
            # args[2][0].set(ary=f['u'][:], queue=args[2][1])
            # args[3][0].set(ary=f['B'][:], queue=args[2][1])

            return f['t'][0], f['rho'][:], f['p'][:], f['u'][:], f['B'][:]

    def get_or_create_dir(self, path: Path) -> Path:
        abs_path = Path(f"{self.dir_name}") / path
        if not abs_path.exists():
            abs_path.mkdir(parents=True, exist_ok=True)
        return abs_path

    def save_energy(self, args):
        path = self.get_or_create_dir("energy/")
    
        if not Path(path / "e_kin.hdf5").exists() or self.rw_energy:
            with h5py.File(path / "e_kin.hdf5", "w") as f:
                dset = f.create_dataset("energy", len(args[0]), dtype=np.float64)
                dset[:] = args[0]
                dset = f.create_dataset("time", len(args[2]), dtype=np.float64)
                dset[:] = args[2]
            with h5py.File(path / "e_mag.hdf5", "w") as f:
                dset = f.create_dataset("energy", len(args[1]), dtype=np.float64)
                dset[:] = args[1]
                dset = f.create_dataset("time", len(args[2]), dtype=np.float64)
                dset[:] = args[2]
        else:
            e_kin = None
            e_mag = None
            time = None

            with h5py.File(path / "e_kin.hdf5", "r") as f:
                e_kin = f["energy"][()]
                e_kin = np.append(e_kin, args[0])
                time = f["time"][()]
                time = np.append(time, args[2])

            with h5py.File(path / "e_kin.hdf5", "w") as f:
                dset = f.create_dataset("energy", len(e_kin), dtype=np.float64)
                dset[:] = e_kin
                dset = f.create_dataset("time", len(time), dtype=np.float64)
                dset[:] = time

            with h5py.File(path / "e_mag.hdf5", "r") as f:
                e_mag = f["energy"][()]
                e_mag = np.append(e_mag, args[1])

            with h5py.File(path / "e_mag.hdf5", "w") as f:
                dset = f.create_dataset("energy", len(e_mag), dtype=np.float64)
                dset[:] = e_mag
                dset = f.create_dataset("time", len(time), dtype=np.float64)
                dset[:] = time

    def get_energy(self):
        path = self.get_or_create_dir("energy/")

        with h5py.File(path / "e_kin.hdf5", "r") as f:
            kin_e = f['energy'][()]

        with h5py.File(path / "e_mag.hdf5", "r") as f:
            mag_e = f['energy'][()]

        return kin_e, mag_e
