import json
import math
from pathlib import Path

class Config:
    file_path = None
    # debug
    debug = False
    # count of ghost points
    ghosts = []
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
    # initials = ""
    # similarity numbers
    RHO0 = 1e0
    U0 = 1e0
    B0 = 1e0

    Re = 1e3
    Rem = 1e3
    delta_hall = 1e1
    Ma = 1e0
    Ms = 1e0
    gamma = 7.0/5.0

    ideal = False
    hall = False
    
    CFL = 0.9

    models = ['DNS', 'SMAGORINSKY', 'CROSS_HELICITY']
    initials = ['OT', 'RAND']

    def __init__(self, file_path=Path.cwd() / "config.json"):
        self.file_path = file_path

        if not self.file_path.exists():
            self.generate_default()

        self._read_file()

    def generate_default(self):
        str = """
{
    "ghosts": [3, 3, 3], 
    "shape": [64, 64, 64], 
    "size": [6.283185307179586, 6.283185307179586, 6.283185307179586], 
    "rw_delemiter": 10,
    "end_time": 0.1,
    "start_step": 0,
    "model": "DNS",
    "initials": "OT",

    "RHO0": 1.0,
    "U0": 1.0,
    "B0": 1.0,

    "Re": 1000.0,
    "Rem": 100.0,
    "delta_hall": 1.0,
    "Ma": 1.0,
    "Ms": 1.0,
    "gamma": 1.4,

    "ideal": false,
    "hall": false,
    "CFL": 0.9
}
        """

        with open(self.file_path, "w") as f:
            print(str, file=f)

    def _read_file(self):
        with open(self.file_path, 'r') as f:
            data = dict(json.load(f))

            self.ghosts = data.get('ghosts', [3, 3, 3])

            self.true_shape = tuple(data.get('shape', (0, 0, 0,)))

            self.true_v_shape = (3, ) + self.true_shape
            self.true_mat_shape = (3, 3, ) + self.true_shape

            self.domain_size = tuple(data.get('size', (1.0, 1.0, 1.0,)))

            self.dV = 1
            
            for i in range(3):
                self.dV *= self.domain_size[i] / self.true_shape[i]

            self.shape = (self.true_shape[0]+2*self.ghosts[0], 
                        self.true_shape[1]+2*self.ghosts[1], 
                        self.true_shape[2]+2*self.ghosts[2], )
            
            self.v_shape = (3, ) + self.shape
            self.mat_shape = (3, 3, ) + self.shape
            
            self.end_time = data.get('end_time', 0)

            self.rw_del = data.get('rw_delemiter', 1)

            self.start_step = data.get('start_step', 0)

            self.model = data.get('model', 'DNS')

            self.initials = data.get("initials", "ot")

            self.RHO0 = data.get("RHO0", 1.0)
            self.U0 = data.get("U0", 1.0)
            self.B0 = data.get("B0", 1.0)

            self.Re = data.get("Re", 1000)
            self.Rem = data.get("Rem", 100)
            self.delta_hall = data.get("delta_hall", 1.0)
            self.Ma = data.get("Ma", 1.0)
            self.Ms = data.get("Ms", 0.2)
            self.gamma = data.get("gamma", (7.0/5.0))

            self.ideal = data.get("ideal", False)
            self.hall = data.get("hall", False)

            self.CFL = data.get("CFL", 0.9)

            # self._generate_defines()
            
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
