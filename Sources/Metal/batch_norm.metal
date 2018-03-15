//
//  batch_norm.metal
//  MetalBender
//
//  Created by Mathias Claassen on 2/12/18.
//

#include <metal_stdlib>
using namespace metal;

kernel void batch_norm_3(texture2d<half, access::read> inTexture [[texture(0)]],
                         texture2d<half, access::write> outTexture [[texture(1)]],
                         constant half4* params[[buffer(0)]],
                         ushort2 gid [[thread_position_in_grid]]) {
    // params is a buffer with [mean, variance, scale, offset, epsilon]
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    half4 i = inTexture.read(gid);
    half epsilon = params[4].x;
    half4 out = params[2] * (i - params[0]) / sqrt(params[1] + epsilon) + params[3];
    outTexture.write(out, gid);
}

kernel void batch_norm(texture2d_array<half, access::read> inTexture [[texture(0)]],
                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                       constant half4* params[[buffer(0)]],

                       ushort3 gid [[thread_position_in_grid]]
                       ) {
    // params is a buffer with [mean, variance, scale, offset, epsilon]
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() || gid.z >= inTexture.get_array_size()) {
        return;
    }
    half4 i = inTexture.read(ushort2(gid.x, gid.y), gid.z);
    ushort depth = inTexture.get_array_size();
    ushort varianceIndex = depth;
    ushort scaleIndex = 2 * depth;
    ushort offsetIndex = 3 * depth;
    half epsilon = params[4 * depth].x;
    half4 out = params[scaleIndex + gid.z] * (i - params[gid.z]) / sqrt(params[varianceIndex + gid.z] + epsilon)
        + params[offsetIndex + gid.z];
    outTexture.write(out, ushort2(gid.x, gid.y), gid.z);
}
