import os
import pyopencl as cl
from solvers import MHDSolver

def main():
    ctx = cl.create_some_context()
    solver = MHDSolver(ctx)

    solver.solve()


if __name__ == "__main__":
    os.environ['PYOPENCL_CTX'] = '1'
    
    main()
