
#include <metal_stdlib>
using namespace metal;

constant float imageScale [[ function_constant(0) ]];
constant float imageShift [[ function_constant(1) ]];

/// Sums two texture arrays of identical size element-wise
kernel void sum_matrix(texture2d_array<float, access::read> texA [[texture(0)]],
                       texture2d_array<float, access::read> texB [[texture(1)]],
                       texture2d_array<float, access::write> outTexture [[texture(2)]],
                       ushort3 gid [[thread_position_in_grid]]) {

    const half4 a = half4(texA.read(ushort2(gid.x, gid.y), gid.z));
    const half4 b = half4(texB.read(ushort2(gid.x, gid.y), gid.z));
    outTexture.write(float4(a + b), ushort2(gid.x, gid.y), gid.z);
}

kernel void sum_matrix_3(texture2d<float, access::read> texA [[texture(0)]],
                         texture2d<float, access::read> texB [[texture(1)]],
                         texture2d<float, access::write> outTexture [[texture(2)]],
                         ushort2 gid [[thread_position_in_grid]]) {

    const half4 a = half4(texA.read(gid));
    const half4 b = half4(texB.read(gid));
    outTexture.write(float4(a + b), gid);
}

/// Transforms BGRA to RGBA
kernel void bgra_to_rgba(
                        texture2d<float, access::read> inTexture [[texture(0)]],
                        texture2d<float, access::write> outTexture [[texture(1)]],
                        ushort2 gid [[thread_position_in_grid]]
                        ) {
    float4 i = inTexture.read(gid);
    outTexture.write(float4(i.z, i.y, i.x, 0.0), gid);
}

/// Applies a scale and shift and clamps between 0 and 1
kernel void image_linear_transform(texture2d<float, access::read> inTexture [[texture(0)]],
                                   texture2d<float, access::write> outTexture [[texture(1)]],
                                   ushort2 gid [[thread_position_in_grid]]
                                   ) {
    half4 i = half4(inTexture.read(gid));
    half4 out = clamp(i*imageScale + imageShift, 0.0h, 1.0h);
    outTexture.write(float4(out.r, out.g, out.b, 1.0), ushort2(gid.x,gid.y));
}
