//
//  instanceNorm3.metal
//  Palladium
//
//  Created by Mathias Claassen on 11/30/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void instance_norm_3(constant float4* weights[[buffer(0)]],
                            constant float4* bias[[buffer(1)]],
                            texture2d<float, access::read> in[[texture(0)]],
                            texture2d<float, access::write> out[[texture(1)]],
                            ushort2 gid[[thread_position_in_grid]],
                            ushort tid[[thread_index_in_threadgroup]],
                            ushort2 tg_size[[threads_per_threadgroup]]) {
    constexpr ushort THREADGROUP_SIZE = 256;

    threadgroup float4 per_thread_state[THREADGROUP_SIZE];
    // Each block handles a single texture.
    per_thread_state[tid] = 0;
    for (ushort y = gid.y; y < in.get_height(); y += tg_size.y) {
        for (ushort x = gid.x; x < in.get_width(); x += tg_size.x) {
            per_thread_state[tid] += in.read(ushort2(x, y));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 256 -> 32 reduction
    if (tid < 32) {
        for (ushort i = tid + 32; i < THREADGROUP_SIZE; i += 32) {
            per_thread_state[tid] += per_thread_state[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float4 sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const float4 mean = per_thread_state[0];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    per_thread_state[tid] = 0;
    for (ushort y = gid.y; y < in.get_height(); y += tg_size.y) {
        for (ushort x = gid.x; x < in.get_width(); x += tg_size.x) {
            float4 delta = in.read(ushort2(x, y)) - mean;
            per_thread_state[tid] += delta * delta;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 256 -> 32 reduction
    if (tid < 32) {
        for (ushort i = tid + 32; i < THREADGROUP_SIZE; i += 32) {
            per_thread_state[tid] += per_thread_state[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float4 sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = 1.0 / sqrt(max(sum, float4(1e-4)) + 1.0e-4);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const float4 inv_var = per_thread_state[0];

    const float4 scale = inv_var * weights[0];
    const float4 shift = bias[0] - mean * scale;

    for (ushort y = gid.y; y < in.get_height(); y += tg_size.y) {
        for (ushort x = gid.x; x < in.get_width(); x += tg_size.x) {
            float4 scaled = in.read(ushort2(x, y)) * scale + shift;
            out.write(clamp(scaled, -10.0, 10.0), ushort2(x, y));
        }
    }
}
