//
//  instanceNorm3.metal
//  Bender
//
//  Adapted from Caffe2 at https://github.com/caffe2/caffe2/blob/d0ce496d2fdf9c0d0ded73f8552e18a82a85e1ba/caffe2/contrib/mpscnn-fb/MPSCNN.metal#L185-L275 with license found at LICENSE_CAFFE2
//  Adapted by Mathias Claassen.
//  Copyright © 2017 Xmartlabs. All rights reserved.

#include <metal_stdlib>
using namespace metal;

kernel void instance_norm_3(constant float4 &scale[[buffer(0)]],
                            constant float4 &shift[[buffer(1)]],
                            texture2d<float, access::read> in[[texture(0)]],
                            texture2d<float, access::write> out[[texture(1)]],
                            ushort2 gid[[thread_position_in_grid]],
                            ushort tid[[thread_index_in_threadgroup]],
                            ushort2 tg_size[[threads_per_threadgroup]]) {
    ushort width = in.get_width();
    ushort height = in.get_height();
    const ushort thread_count = tg_size.x * tg_size.y;
    threadgroup float4 shared_mem [256];

    float4 sum = 0;
    for(ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for(ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            sum += in.read(ushort2(xIndex, yIndex));
        }
    }
    shared_mem[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to 32 values
    sum = 0;
    if (tid < 32) {
        for (ushort i = tid + 32; i < thread_count; i += 32) {
            sum += shared_mem[i];
        }
    }
    shared_mem[tid] += sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate mean
    sum = 0;
    if (tid == 0) {
        ushort top = min(ushort(32), thread_count);
        for (ushort i = 0; i < top; i += 1) {
            sum += shared_mem[i];
        }
        shared_mem[0] = sum / (width * height);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 mean = shared_mem[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Variance
    sum = 0;
    for(ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for(ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            sum += pow(in.read(ushort2(xIndex, yIndex)) - mean, 2);
        }
    }
    shared_mem[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce to 32 values
    sum = 0;
    if (tid < 32) {
        for (ushort i = tid + 32; i < thread_count; i += 32) {
            sum += shared_mem[i];
        }
    }
    shared_mem[tid] += sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Calculate variance
    sum = 0;
    if (tid == 0) {
        ushort top = min(ushort(32), thread_count);
        for (ushort i = 0; i < top; i += 1) {
            sum += shared_mem[i];
        }
        shared_mem[0] = sum / (width * height);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 sigma = sqrt(shared_mem[0] + float4(1e-4));

    float4 multiplier = scale / sigma;
    for(ushort xIndex = gid.x; xIndex < width; xIndex += tg_size.x) {
        for(ushort yIndex = gid.y; yIndex < height; yIndex += tg_size.y) {
            float4 val = in.read(ushort2(xIndex, yIndex));
            out.write(clamp((val - mean) * multiplier + shift, -10.0, 10.0), ushort2(xIndex, yIndex));
        }
    }
}
