//
//  ColorShaders.metal
//  Palladium
//
//  Created by Mathias Claassen on 4/25/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

half3 grayScale(half3 rgb);
half3 RGB_to_YCbCr(half3 rgb);
half3 YCbCr_to_RGB(half3 ycbcr);

constant half3 kRec601Luma = half3(0.299, 0.587, 0.114);
constant half3 YCbCrOffset = half3((16.0/255.0), 0.5, 0.5);


// MARK: Luminance

kernel void luminance_transfer(
                            texture2d<float, access::read> luminanceTexture [[texture(0)]],
                            texture2d<half, access::read> colorTexture [[texture(1)]],
                            texture2d<float, access::write> outTexture [[texture(2)]],

                            ushort2 gid [[thread_position_in_grid]]
                            ) {

    half4 lumi = half4(luminanceTexture.read(gid));
    half4 color = half4(colorTexture.read(gid));

    half3 lumiYUV = RGB_to_YCbCr(grayScale(lumi.rgb));
    half3 colorYUV = RGB_to_YCbCr(color.rgb);

    half3 outRGB = YCbCr_to_RGB(half3(lumiYUV.r, colorYUV.g, colorYUV.b));
    outTexture.write(float4(half4(outRGB, 1.0h)), gid);
}

half3 grayScale(half3 rgb)
{
    half gray = dot(rgb, kRec601Luma);
    return half3(gray);
}

half3 RGB_to_YCbCr(half3 rgb)
{
    half3x3 RGBToYuvMatrix = half3x3(half3(0.257h, -0.148h,  0.439h),
                                     half3(0.504h, -0.291h, -0.368h),
                                     half3(0.098h,  0.439h, -0.071h));

    return (RGBToYuvMatrix * rgb) + YCbCrOffset;
}

half3 YCbCr_to_RGB(half3 ycbcr)
{

    half3x3 yuvToRGBMatrix = half3x3(half3(1.164,  1.164, 1.164),
                                     half3(0.000, -0.392, 2.017),
                                     half3(1.596, -0.813, 0.000));

    return yuvToRGBMatrix * (ycbcr - YCbCrOffset);
}

// MARK: GrayScale

kernel void to_grayscale(texture2d<float, access::read> inTexture [[texture(0)]],
                         texture2d<float, access::write> outTexture [[texture(1)]],

                         ushort2 gid [[thread_position_in_grid]]
                         ) {

    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }

    half4 input = half4(inTexture.read(gid));
    float gray = float(grayScale(input.rgb).r);
    outTexture.write(float4(gray, gray, gray, 1.0), gid);
//    outTexture.write(inTexture.read(gid), gid);

}
