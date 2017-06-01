//
//  instanceNorm.metal
//  Palladium
//
//  Created by Mathias Claassen on 11/30/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void meanA(
                  texture2d_array<float, access::read> src [[texture(0)]],
                  texture2d_array<float, access::write> outTexture [[texture(1)]],
                  ushort2 gid [[thread_position_in_grid]])
{
    half4 sum = half4(0.0h);
    half4 blockSum = half4(0.0h);
    const ushort blockSize = 16;
    const ushort size = src.get_width();
    const ushort blocks = size/blockSize;
    
    for (ushort b=0; b<blocks; b++){
        for (ushort i=0; i<blockSize; i++)
        {
            const half4 color = half4(src.read(ushort2(gid.x, i+b*blockSize), gid.y));
            blockSum += color;
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }

    outTexture.write(float4(sum / half(blocks)), ushort2(gid.x, 0), gid.y);
}

kernel void avgMean(
                    texture2d_array<float, access::read> src [[texture(0)]],
                    texture2d_array<float, access::write> outTexture [[texture(1)]],

                    ushort2 gid [[thread_position_in_grid]])
{
    half4 sum = half4(0.0h);
    half4 blockSum = half4(0.0h);
    const ushort blockSize = 16;
    const ushort size = src.get_width();
    const ushort blocks = size/blockSize;
    
    for (ushort b=0; b<blocks; b++){
        for (ushort i=0; i<blockSize; i++)
        {
            const half4 color = half4(src.read(ushort2(i+b*blockSize, 0), gid.x));
            blockSum += color;
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }
    outTexture.write(float4(sum / half(blocks)), ushort2(0,0), gid.x);
}

kernel void varianceA(
                      texture2d_array<float, access::read> src [[texture(0)]],
                      texture2d_array<float, access::read> meanTexture [[texture(1)]],
                      texture2d_array<float, access::write> outTexture [[texture(2)]],

                      ushort2 gid [[thread_position_in_grid]])
{
    const half4 mu = half4(meanTexture.read(ushort2(0, 0), gid.y));
    
    half4 sum = half4(0.0h);
    half4 blockSum = half4(0.0h);
    const ushort blockSize = 16;
    const ushort size = src.get_width();
    const ushort blocks = size/blockSize;
    
    for (ushort b=0; b<blocks; b++){
        for (ushort i=0; i<blockSize; i++)
        {
            const half4 val = half4(src.read(ushort2(gid.x, i+b*blockSize), gid.y));
            blockSum += (val - mu) * (val - mu);
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }
    outTexture.write(float4(sum / half(blocks)), ushort2(gid.x, 0), gid.y);
}

kernel void avgVar(
                   texture2d_array<float, access::read> src [[texture(0)]],
                   texture2d_array<float, access::write> outTexture [[texture(1)]],

                   ushort2 gid [[thread_position_in_grid]])
{
    half4 sum = half4(0.0h);
    half4 blockSum = half4(0.0h);
    const ushort blockSize = 16;
    const ushort size = src.get_width();
    const ushort blocks = size/blockSize;
    
    for (ushort b=0; b<blocks; b++){
        for (ushort i=0; i<blockSize; i++)
        {
            const half4 color = half4(src.read(ushort2(i+b*blockSize, 0), gid.x));
            blockSum += color;
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }
    const half4 sigma = sum / half(blocks);

    outTexture.write(float4(sqrt(sigma + 0.001h)), ushort2(0, 0), gid.x);
}

kernel void instanceNorm(
                         texture2d_array<float, access::read> src [[texture(0)]],
                         texture2d_array<float, access::read> meanTexture [[texture(1)]],
                         texture2d_array<float, access::read> varianceTexture [[texture(2)]],
                         texture2d_array<float, access::write> outTexture [[texture(3)]],
                         
                         constant float4 *scale [[ buffer(0) ]],
                         constant float4 *shift [[ buffer(1) ]],

                         ushort3 gid [[thread_position_in_grid]])
{
    const half4 val = half4(src.read(ushort2(gid.x, gid.y), gid.z));
    const half4 mu = half4(meanTexture.read(ushort2(0, 0), gid.z));
    const half4 sigma = half4(varianceTexture.read(ushort2(0, 0), gid.z));
    
    outTexture.write(clamp(float4((val - mu) / sigma) * scale[gid.z] + shift[gid.z], -10.0, 10.0), ushort2(gid.x, gid.y), gid.z);
}
