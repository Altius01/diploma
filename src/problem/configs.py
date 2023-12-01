from dataclasses import dataclass
from typing import List

from common.config import Config
from src.common.types import *
from src.common.matrix_ops import *


@dataclass
class SystemConfig:
    CFL: float

    rk_steps: int

    initials: str
    les_model: str

    hall: bool
    ideal: bool

    h: List[float]

    shape: List[int]
    ghosts: List[int]
    domain_size: List[int]

    RHO0: float
    U0: float
    B0: float
    eps_p: float

    gamma: float
    Re: float
    Ms: float
    Ma: float
    Rem: float
    nu_0: float
    delta_hall: float

    rw_del: int
    end_time: float
    start_step: int

    def __init__(self, config: Config):
        self.RHO0 = config.RHO0
        self.U0 = config.U0
        self.B0 = config.B0
        self.eps_p = 1e-5
        self.h = [0, 0, 0]
        self.Re = config.Re
        self.nu_0 = config.nu_0
        self.Rem = config.Rem
        self.delta_hall = config.delta_hall
        self.Ma = config.Ma
        self.Ms = config.Ms
        self.gamma = config.gamma

        self.CFL = config.CFL

        self.rw_del = config.rw_del

        self.end_time = config.end_time
        self.start_step = config.start_step

        self.rk_steps = 2
        self.les_model = NonHallLES(config.model)
        self.initials = Initials(config.initials)
        self.ideal = config.ideal
        self.hall = config.hall

        self.dim = config.dim

        self.domain_size = config.domain_size

        self.debug_fv_step = True

        self.config = config
        self.ghosts = config.ghosts
        self.shape = config.shape
        self.true_shape = config.true_shape


@dataclass
class ProblemConfig:
    gamma: double
    Re: double
    Ms: double
    Ma: double
    Rem: double
    delta_hall: double

    ghosts: int

    shape: tuple
    h: tuple
    domain_size: tuple

    ideal: bool
    hall: bool

    dim: int

    def __init__(self, sys_cfg: SystemConfig):
        self.gamma = sys_cfg.gamma
        self.Re = sys_cfg.Re
        self.Ms = sys_cfg.Ms
        self.Ma = sys_cfg.Ma
        self.Rem = sys_cfg.Rem
        self.delta_hall = sys_cfg.delta_hall
        self.ghost = sys_cfg.ghosts
        self.config = sys_cfg.config
        self.h = sys_cfg.h
        self.config = sys_cfg.config
        self.ideal = sys_cfg.ideal
        self.hall = sys_cfg.hall
        self.dim = sys_cfg.dim

        self.les = sys_cfg.les_model

        self.filter_size = np.array([1, 1, 1])
