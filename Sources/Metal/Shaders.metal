
#include <metal_stdlib>
using namespace metal;

kernel void sum_matrix(texture2d_array<float, access::read> texA [[texture(0)]],
                       texture2d_array<float, access::read> texB [[texture(1)]],
                       texture2d_array<float, access::write> outTexture [[texture(2)]],
                       ushort3 gid [[thread_position_in_grid]]) {

    const half4 a = half4(texA.read(ushort2(gid.x, gid.y), gid.z));
    const half4 b = half4(texB.read(ushort2(gid.x, gid.y), gid.z));
    outTexture.write(float4(a + b), ushort2(gid.x, gid.y), gid.z);
}

kernel void bgra_to_rgba(
                        texture2d<float, access::read> inTexture [[texture(0)]],
                        texture2d<float, access::write> outTexture [[texture(1)]],
                        ushort2 gid [[thread_position_in_grid]]
                        ) {
    float4 i = inTexture.read(gid);
    outTexture.write(float4(i.z, i.y, i.x, 0.0), gid);
}

kernel void scale_to_float(
                           texture2d<float, access::read> inTexture [[texture(0)]],
                           texture2d<float, access::write> outTexture [[texture(1)]],
                           ushort2 gid [[thread_position_in_grid]]
                           ) {
    half4 i = half4(inTexture.read(gid));
    half r = clamp((i.x+1.0h)*0.5h, 0.0h, 1.0h);
    half g = clamp((i.y+1.0h)*0.5h, 0.0h, 1.0h);
    half b = clamp((i.z+1.0h)*0.5h, 0.0h, 1.0h);
    outTexture.write(float4(r, g, b, 1.0), ushort2(gid.x,gid.y));
}

kernel void apply_watermark(
                            texture2d<float, access::read> inTexture [[texture(0)]],
                            texture2d<half, access::read> watermark [[texture(1)]],
                            texture2d<float, access::write> outTexture [[texture(2)]],

                            ushort2 gid [[thread_position_in_grid]]
                            ) {
    half4 i = half4(inTexture.read(gid));
    ushort wWidth = watermark.get_width();
    ushort wHeight = watermark.get_height();
    ushort xOffset =  inTexture.get_width() - 10 - wWidth;
    ushort yOffset = inTexture.get_height() - wHeight - 10;

    if (gid.x > xOffset && gid.x < xOffset + wWidth && gid.y > yOffset && gid.y < yOffset + wHeight) {
        half4 w = half4(watermark.read(ushort2(gid.x - xOffset, gid.y - yOffset)));

        half r = clamp((i.x+1.0h)*0.5h * (1.0h-w.a) + w.a*w.x, 0.0h, 1.0h); //
        half g = clamp((i.y+1.0h)*0.5h * (1.0h-w.a) + w.a*w.y, 0.0h, 1.0h); //
        half b = clamp((i.z+1.0h)*0.5h * (1.0h-w.a) + w.a*w.z, 0.0h, 1.0h); //
        outTexture.write(float4(r, g, b, 1.0), ushort2(gid.x,gid.y));
    } else {
        outTexture.write(float4(i.r, i.g, i.b, 1.0), ushort2(gid.x,gid.y));
    }
}

