//
//  local_response_norm.metal
//  Bender
//
//  Created by Diego Ernst on 5/22/17.
//
//

#include <metal_stdlib>
using namespace metal;

constant ushort depthRadius   [[ function_constant(0) ]];
constant float biasParam      [[ function_constant(1) ]];
constant float alpha          [[ function_constant(2) ]];
constant float beta           [[ function_constant(3) ]];

#define DECLARE_UNION(name, s)\
union name ## _t {\
    half4 block[(s)] = {0};\
    half array[(s) * 4];\
} name;\


#define COMPUTE(array, size, offset)\
union result_t {\
    half4 value;\
    half  arrayy[4] = {0};\
} result;\
\
for(auto z = (offset); z < (offset) + 4; z++) {\
    auto input = array[z];\
    float sqrSum = 0;\
    const ushort maxZ = z + depthRadius >= (size) ? ((size) - 1) : z + depthRadius;\
    for (ushort zRadius = max(z - depthRadius, 0); zRadius <= maxZ; zRadius++) {\
        half zRadiusValue = array[zRadius];\
        sqrSum += pow(zRadiusValue, 2);\
    }\
\
    float divisor = pow(biasParam + alpha * sqrSum, beta);\
    result.arrayy[z - (offset)] = input / divisor;\
};\
dest.write(float4(result.value), ushort2(gid.x, gid.y), gid.z);\


#define FILL_BLOCKS(block, count, maxBlocks)\
ushort middle = ((count) - 1) / 2;\
for (ushort b = 0; b < (count); b++){\
    short gidz = gid.z + (b - middle);\
    if (gidz >= 0 && ((ushort)gidz < (maxBlocks))) {\
        block[b] = half4(src.read(ushort2(gid.x, gid.y), gidz));\
    }\
}\


#define COMPUTE_LOCAL_RESPONSE_NORM(offsetChunks, maxBlocks)\
DECLARE_UNION(buffer, (offsetChunks) * 2 + 1)\
FILL_BLOCKS(buffer.block, (offsetChunks) * 2 + 1, maxBlocks)\
COMPUTE(buffer.array, ((offsetChunks) * 2 + 1) * 4, (offsetChunks) * 4)

/// Implements Local Response Noramlization for texture arrays
kernel void local_response_norm(
                            texture2d_array<float, access::read> src [[texture(0)]],
                            texture2d_array<float, access::write> dest [[texture(1)]],
                            ushort3 gid [[thread_position_in_grid]]) {

    // assumes depthRadius <= 20

    const ushort offsetChunks = depthRadius & 0x03 ? (depthRadius >> 2) + 1 : (depthRadius >> 2);

    if (offsetChunks <= 1) {

        COMPUTE_LOCAL_RESPONSE_NORM(1, src.get_array_size());

    } else if (offsetChunks == 2) {

        COMPUTE_LOCAL_RESPONSE_NORM(2, src.get_array_size());

    } else if (offsetChunks == 3) {

        COMPUTE_LOCAL_RESPONSE_NORM(3, src.get_array_size());
        
    } else if (offsetChunks == 4) {

        COMPUTE_LOCAL_RESPONSE_NORM(4, src.get_array_size());

    } else if (offsetChunks == 5) {

        COMPUTE_LOCAL_RESPONSE_NORM(5, src.get_array_size());
        
    }

}

/// Implements Local Response Noramlization for simple textures
kernel void local_response_norm_3(
                              texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dest [[texture(1)]],
                              ushort3 gid [[thread_position_in_grid]]) {

    COMPUTE_LOCAL_RESPONSE_NORM(0, 1);

}
