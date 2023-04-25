import json
import math
from pathlib import Path

class Config:
    file_path = None
    # debug
    debug = False
    # count of ghost points
    ghosts = 0
    # geometry
    shape = ()
    v_shape = ()
    mat_shape=()
    true_shape = ()
    true_v_shape = ()
    true_mat_shape = ()
    domain_size = ()
    dV = 0
    # end_time
    t_end = 0
    # steps count
    steps = 0
    # start step
    start_step = 0
    # saving freq
    rw_del = 0
    rewrite_energy = True
    defines = []
    initials = ""

    models = ['DNS', 'SMAGORINSKY', 'CROSS_HELICITY']

    def __init__(self, file_path=Path.cwd() / "config.json"):
        self.file_path = file_path

        if not self.file_path.exists():
            self.generate_default()

        self._read_file()

    def generate_default(self):
        str = """
{
    "ghosts": 3, 
    "shape": [64, 64, 64], 
    "size": [6.283185307179586, 6.283185307179586, 6.283185307179586], 
    "rw_delemiter": 10,
    "end_time": 0.1,
    "start_step": 0,
    "model": "DNS"
}
        """

        with open(self.file_path, "w") as f:
            print(str, file=f)

    def _read_file(self):
        with open(self.file_path, 'r') as f:
            data = dict(json.load(f))

            self.ghosts = data.get('ghosts', 2)

            self.true_shape = tuple(data.get('shape', (0, 0, 0,)))

            self.true_v_shape = (3, ) + self.true_shape
            self.true_mat_shape = (3, 3, ) + self.true_shape

            self.domain_size = tuple(data.get('size', (1.0, 1.0, 1.0,)))

            self.dV = 1
            
            for i in range(3):
                self.dV *= self.domain_size[i] / self.true_shape[i]

            self.shape = (self.true_shape[0]+2*self.ghosts, 
                        self.true_shape[1]+2*self.ghosts, 
                        self.true_shape[2]+2*self.ghosts, )
            
            self.v_shape = (3, ) + self.shape
            self.mat_shape = (3, 3, ) + self.shape
            
            self.end_time = data.get('end_time', 0)

            self.rw_del = data.get('rw_delemiter', 1)

            self.start_step = data.get('start_step', 0)

            self.model = data.get('model', 'DNS')

            self.initials = data.get("initials", "ot")

            self._generate_defines()
            
            if self.start_step == 0:
                self.rewrite_energy = True
            else:
                self.rewrite_energy = False

    def _generate_defines(self):
        self.defines = []
        for model in self.models:
            model_flag = "false"
            if self.model.lower() == model.lower():
                model_flag = "true"
            self.defines.append([model, model_flag])

        self.defines.append(['GHOST_CELLS', self.ghosts])

        self.defines.append(['TRUE_Nx', self.true_shape[0]])
        self.defines.append(['TRUE_Ny', self.true_shape[1]])
        self.defines.append(['TRUE_Nz', self.true_shape[2]])
