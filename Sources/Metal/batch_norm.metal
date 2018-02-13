//
//  batch_norm.metal
//  MetalBender
//
//  Created by Mathias Claassen on 2/12/18.
//

#include <metal_stdlib>
using namespace metal;

kernel void batch_norm_3(texture2d<float, access::read> inTexture [[texture(0)]],
                         texture2d<float, access::write> outTexture [[texture(1)]],
                         constant float4* params[[buffer(0)]],
                         ushort2 gid [[thread_position_in_grid]]) {
    
    float4 i = inTexture.read(gid);
    float4 out = params[2] * (i - params[0]) / params[1] + params[3];
    outTexture.write(out, gid);
}

kernel void batch_norm(texture2d_array<float, access::read> inTexture [[texture(0)]],
                       texture2d_array<float, access::write> outTexture [[texture(1)]],
                       constant float4* params[[buffer(0)]],

                       ushort3 gid [[thread_position_in_grid]]
                       ) {

    float4 i = inTexture.read(ushort2(gid.x, gid.y), gid.z);
    int depth = inTexture.get_array_size();
    float4 out = params[2 * depth + gid.z] * (i - params[gid.z]) / params[depth + gid.z] + params[3 * depth + gid.z];
    outTexture.write(float4(out), ushort2(gid.x, gid.y), gid.z);
}
