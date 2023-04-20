import re
from pathlib import Path

import pyopencl as cl

from config import Config
from logger import Logger

#TODO: move to config file
source_dir=Path(__file__).parent / './c_sources'

class CLBuilder():
    def build(context, defines):
        Logger.log(f"Start building sources from: {source_dir}")

        with open((source_dir / 'main.cl'), 'r') as file:
            data = file.read()
            for name, value in defines:
                data = CLBuilder._replace_define(data, name, value)

        program = cl.Program(context, data).build(options=['-I', 
                                                            str(source_dir)])

        Logger.log(f"Building sources from: {source_dir} - done!")
        return program

    def _replace_define(src, define_name, value):
        reg_str = f'#define {define_name} [a-zA-Z0-9_.-]*'
        replace_str = f'#define {define_name} {value}'
        
        return re.sub(reg_str, replace_str, src)
    