//
//  instanceNorm3.metal
//  VideoStylizer
//
//  Created by Mathias Claassen on 11/30/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void meanA_3(
                  texture2d<float, access::read> src [[texture(0)]],
                  texture2d<float, access::write> outTexture [[texture(1)]],

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
            const half4 color = half4(src.read(ushort2(gid.x, i+b*blockSize)));
            blockSum += color;
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }

    outTexture.write(float4(sum / half(blocks)), gid);
}

kernel void avgMean_3(
                    texture2d<float, access::read> src [[texture(0)]],
                    texture2d<float, access::write> outTexture [[texture(1)]],

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
            const half4 color = half4(src.read(ushort2(i+b*blockSize, 0)));
            blockSum += color;
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }
    outTexture.write(float4(sum / half(blocks)), ushort2(0,0));
}

kernel void varianceA_3(
                      texture2d<float, access::read> src [[texture(0)]],
                      texture2d<float, access::read> meanTexture [[texture(1)]],
                      texture2d<float, access::write> outTexture [[texture(2)]],

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
            const half4 val = half4(src.read(ushort2(gid.x, i+b*blockSize)));
            const half4 mu = half4(meanTexture.read(ushort2(0, 0)));
            blockSum += (val - mu) * (val - mu);
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }
    outTexture.write(float4(sum / half(blocks)), gid);
}

kernel void avgVar_3(
                   texture2d<float, access::read> src [[texture(0)]],
                   texture2d<float, access::write> outTexture [[texture(1)]],

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
            const half4 color = half4(src.read(ushort2(i+b*blockSize, 0)));
            blockSum += color;
        }
        sum += blockSum / half(blockSize);
        blockSum = half4(0.0h);
    }
    const half4 sigma = sum / half(blocks);

    outTexture.write(float4(sqrt(sigma + 0.001h)), ushort2(0, 0));
}

kernel void instanceNorm_3(
                         texture2d<float, access::read> src [[texture(0)]],
                         texture2d<float, access::read> meanTexture [[texture(1)]],
                         texture2d<float, access::read> varianceTexture [[texture(2)]],
                         texture2d<float, access::write> outTexture [[texture(3)]],
                           
                           constant float4 &scale [[ buffer(0) ]],
                           constant float4 &shift [[ buffer(1) ]],
                           
                         ushort3 gid [[thread_position_in_grid]])
{
    const half4 val = half4(src.read(ushort2(gid.x, gid.y)));
    const half4 mu = half4(meanTexture.read(ushort2(0, 0)));
    const half4 sigma = half4(varianceTexture.read(ushort2(0, 0)));
    
    outTexture.write(clamp(float4((val - mu) / sigma) * scale + shift, -10.0, 10.0), ushort2(gid.x, gid.y));
}
