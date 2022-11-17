from enum import Enum
from datetime import date

class Initial(Enum):
    ORSZAG_TANG = 'O-T'
    

class Solver():

    def __init__(self, steps, dir_name=date.today()) -> None:
        pass