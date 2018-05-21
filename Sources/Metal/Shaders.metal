
#include <metal_stdlib>

using namespace metal;

typedef half texture_type;
typedef half4 texture_type4;
typedef half calculation_type;
typedef half4 calculation_type4;

constant float imageScale [[ function_constant(0) ]];
constant float imageShift [[ function_constant(1) ]];

/// Sums two texture arrays of identical size element-wise
kernel void sum_matrix(texture2d_array<texture_type, access::read> texA [[texture(0)]],
                       texture2d_array<texture_type, access::read> texB [[texture(1)]],
                       texture2d_array<texture_type, access::write> outTexture [[texture(2)]],
                       ushort3 gid [[thread_position_in_grid]]) {

    const calculation_type4 a = calculation_type4(texA.read(ushort2(gid.x, gid.y), gid.z));
    const calculation_type4 b = calculation_type4(texB.read(ushort2(gid.x, gid.y), gid.z));
    outTexture.write(texture_type4(a + b), ushort2(gid.x, gid.y), gid.z);
}

kernel void sum_matrix_3(texture2d<texture_type, access::read> texA [[texture(0)]],
                         texture2d<texture_type, access::read> texB [[texture(1)]],
                         texture2d<texture_type, access::write> outTexture [[texture(2)]],
                         ushort2 gid [[thread_position_in_grid]]) {

    const calculation_type4 a = calculation_type4(texA.read(gid));
    const calculation_type4 b = calculation_type4(texB.read(gid));
    outTexture.write(texture_type4(a + b), gid);
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
kernel void image_linear_transform(texture2d<texture_type, access::read> inTexture [[texture(0)]],
                                   texture2d<texture_type, access::write> outTexture [[texture(1)]],
                                   ushort2 gid [[thread_position_in_grid]]
                                   ) {
    calculation_type4 i = calculation_type4(inTexture.read(gid));
    calculation_type4 out = clamp(i*imageScale + imageShift, 0.0h, 1.0h);
    outTexture.write(texture_type4(out.r, out.g, out.b, 1.0h), ushort2(gid.x,gid.y));
}

/// Multiplies a scalar by an image
kernel void multiply_scalar(texture2d_array<texture_type, access::read> inTexture [[texture(0)]],
                            texture2d_array<texture_type, access::write> outTexture [[texture(1)]],
                            ushort3 gid [[thread_position_in_grid]]
                            ) {
    outTexture.write(inTexture.read(ushort2(gid.x,gid.y), gid.z) * imageScale, ushort2(gid.x,gid.y), gid.z);
}

/// Multiplies a scalar by an image
kernel void multiply_scalar_3(texture2d<texture_type, access::read> inTexture [[texture(0)]],
                              texture2d<texture_type, access::write> outTexture [[texture(1)]],
                              ushort2 gid [[thread_position_in_grid]]
                              ) {
    outTexture.write(inTexture.read(gid) * imageScale, ushort2(gid.x,gid.y));
}
