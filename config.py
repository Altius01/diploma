import json
from pathlib import Path

class Config:
    GHOSTS = 0
    SHAPE = ()
    T_SHAPE = ()
    STEPS = 0
    START_STEP = 0
    RW_DELETIMER = 0
    REWRITE_ENERGY = True 

    def __init__(self, file_path=Path(".") / "config.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)

            self.GHOSTS = data['ghosts']
            self.T_SHAPE = tuple(data['shape'])
            self.SHAPE = (self.T_SHAPE[0]+2*self.GHOSTS, 
                          self.T_SHAPE[1]+2*self.GHOSTS, 
                          self.T_SHAPE[2]+2*self.GHOSTS, )
            self.STEPS = data['steps']
            self.START_STEP = data['start_step']
            self.RW_DELETIMER = data['rw_delemiter']
            self.REWRITE_ENERGY = bool(data['rewrite_energy'])
