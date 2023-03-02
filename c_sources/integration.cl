#ifndef INTEGRATION
#define INTEGRATION

#include "common/mhd_consts.cl"

inline uint global_idx(uint ax, uint x, uint y, uint z) {
        return ax*global_size_x*global_size_y*global_size_z + 
        x * global_size_y*global_size_z +
        y * global_size_z +
        z;
    }

inline uint local_idx(uint x, uint y, uint z) {
    return x*get_local_size(1)*get_local_size(2) + y*get_local_size(2) + z;
}

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

    uint local_id =
        local_idx(local_x, local_y, local_z);

    double3 v = (double3) {u[global_idx(0, global_x, global_y, global_z)], 
                        u[global_idx(1, global_x, global_y, global_z)], 
                        u[global_idx(2, global_x, global_y, global_z)]
    };

    double sq_v = v.x*v.x + v.y*v.y + v.z*v.z;
    double dV = hx*hy*hz;

    // Copy from global to local memory
    localSums[local_id] = 0.5 * rho[global_idx(0, global_x, global_y, global_z)] * sq_v * dV;

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (uint stride_x = group_size_x/2; stride_x>0; stride_x /=2)
    {
        for (uint stride_y = group_size_y/2; stride_y>0; stride_y /=2)
        {
            for (uint stride_z = group_size_z/2; stride_z>0; stride_z /=2)
            {
                // Waiting for each 2x2 addition into given workgroup
                barrier(CLK_LOCAL_MEM_FENCE);

                uint stride_id = local_idx(stride_x, stride_y, stride_z);
                // Add elements 2 by 2 between local_id and local_id + stride
                if (local_x < stride_x && local_y < stride_y && local_z < stride_z)
                    localSums[local_id] += localSums[local_id + stride_id];
            }
        }
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

      uint local_id =
        local_idx(local_x, local_y, local_z);

    double3 v = (double3) {B[global_idx(0, global_x, global_y, global_z)], 
                        B[global_idx(1, global_x, global_y, global_z)], 
                        B[global_idx(2, global_x, global_y, global_z)]
    };

    double sq_v = v.x*v.x + v.y*v.y + v.z*v.z;
    double dV = hx*hy*hz;

    // Copy from global to local memory
    localSums[local_id] = sq_v * dV;

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (uint stride_x = group_size_x/2; stride_x>0; stride_x /=2)
    {
        for (uint stride_y = group_size_y/2; stride_y>0; stride_y /=2)
        {
            for (uint stride_z = group_size_z/2; stride_z>0; stride_z /=2)
            {
                // Waiting for each 2x2 addition into given workgroup
                barrier(CLK_LOCAL_MEM_FENCE);

                uint stride_id = local_idx(stride_x, stride_y, stride_z);
                // Add elements 2 by 2 between local_id and local_id + stride
                if (local_x < stride_x && local_y < stride_y && local_z < stride_z)
                    localSums[local_id] += localSums[local_id + stride_id];
            }
        }
    }

    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
        partialSums[get_group_id(0)] = localSums[0];
}

#endif