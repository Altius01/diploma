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
    true_shape = ()
    true_v_shape = ()
    domain_size = ()
    # end_time
    t_end = 0
    # steps count
    steps = 0
    # start step
    start_step = 0
    # saving freq
    rw_del = 0
    rewrite_energy = True 

    def __init__(self, file_path=Path.cwd() / "config.json"):
        self.file_path = file_path

        if not self.file_path.exists():
            self.generate_default()

        self._read_file()


    def generate_default(self):
        default = {}
        default['ghosts'] = 2
        default['shape'] = (64, 64, 64,)
        default['size'] = (2*math.pi, 2*math.pi, 2*math.pi,)
        default['rw_delemiter'] = 10
        default['steps'] = 100
        default['start_step'] = 0

        with open(self.file_path, "w") as f:
            json.dump(default, f)


    def _read_file(self):
        with open(self.file_path, 'r') as f:
            data = dict(json.load(f))

            self.ghosts = data.get('ghosts', 2)

            self.true_shape = tuple(data.get('shape', (0, 0, 0,)))

            self.true_v_shape = (3, ) + self.true_shape

            self.domain_size = tuple(data.get('size', (1.0, 1.0, 1.0,)))

            self.shape = (self.true_shape[0]+2*self.ghosts, 
                        self.true_shape[1]+2*self.ghosts, 
                        self.true_shape[2]+2*self.ghosts, )
            
            self.v_shape = (3, ) + self.shape
            
            # self.T_END = data.get('end_time', 0)

            self.rw_del = data.get('rw_delemiter', 1)

            self.steps = data.get('steps', 0)

            self.start_step = data.get('start_step', 0)
            
            if self.start_step == 0:
                self.rewrite_energy = True
            else:
                self.rewrite_energy = False
