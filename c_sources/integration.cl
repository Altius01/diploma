#ifndef INTEGRATION
#define INTEGRATION

#include "common/mhd_consts.cl"


__kernel void integrate_kinetic ( __global const double *rho,
                        __global const double *u,
                         __global double *partialSums,
                         __local double *localSums)
{
    uint global_x = get_global_id(0); 
    uint global_y = get_global_id(1); 
    uint global_z = get_global_id(2);

    uint global_size_x = get_local_size(0); 
    uint global_size_y = get_local_size(1); 
    uint global_size_z = get_local_size(2);

    uint local_x = get_local_id(0); 
    uint local_y = get_local_id(1); 
    uint local_z = get_local_id(2);

    uint group_size_x = get_local_size(0); 
    uint group_size_y = get_local_size(1); 
    uint group_size_z = get_local_size(2);

    uint local_idx =
        local_x*group_size_y*group_size_z + local_y*group_size_z + local_z;

    inline uint global_idx(uint ax, uint x, uint y, uint z) {
        return ax*global_size_x*global_size_y*global_size_z + 
        x * global_size_y*global_size_z +
        y * global_size_z +
        z;
    }

    double3 v = (double3) {u[global_idx(0, global_x, global_y, global_z)], 
                        u[global_idx(1, global_x, global_y, global_z)], 
                        u[global_idx(2, global_x, global_y, global_z)]
    };

    double sq_v = v.x*v.x + v.y*v.y + v.z*v.z;
    double dV = hx*hy*hz;

    // Copy from global to local memory
    localSums[local_idx] = 0.5 * rho[global_idx] * sq_v * dV;

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }

    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
        partialSums[get_group_id(0)] = localSums[0];
}

__kernel void integrate_magnetic ( __global const double *B,
                         __global double *partialSums,
                         __local double *localSums)
{
    uint global_x = get_global_id(0); 
    uint global_y = get_global_id(1); 
    uint global_z = get_global_id(2);

    uint global_size_x = get_local_size(0); 
    uint global_size_y = get_local_size(1); 
    uint global_size_z = get_local_size(2);

    uint local_x = get_local_id(0); 
    uint local_y = get_local_id(1); 
    uint local_z = get_local_id(2);

    uint group_size_x = get_local_size(0); 
    uint group_size_y = get_local_size(1); 
    uint group_size_z = get_local_size(2);

    uint local_idx =
        local_x*group_size_y*group_size_z + local_y*group_size_z + local_z;

    inline uint global_idx(uint ax, uint x, uint y, uint z) {
        return ax*global_size_x*global_size_y*global_size_z + 
        x * global_size_y*global_size_z +
        y * global_size_z +
        z;
    }

    double3 v = (double3) {B[global_idx(0, global_x, global_y, global_z)], 
                        B[global_idx(1, global_x, global_y, global_z)], 
                        B[global_idx(2, global_x, global_y, global_z)]
    };

    double sq_v = v.x*v.x + v.y*v.y + v.z*v.z;
    double dV = hx*hy*hz;

    // Copy from global to local memory
    localSums[local_idx] = sq_v * dV;

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }

    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
        partialSums[get_group_id(0)] = localSums[0];
}

#endif