import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from pyfft.cl import Plan

def fftn(arr: np.ndarray, return_np=True):
    assert len(arr.shape) <= 3
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    plan = Plan(arr.shape, queue=queue)
    gpu_data = cl_array.to_device(ctx, queue, arr)
    plan.execute(gpu_data.data)

    if return_np:
        return gpu_data.get()
    else:
        return gpu_data
