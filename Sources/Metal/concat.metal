//
//  concat.metal
//  Bender
//
//  Created by Diego Ernst on 6/27/17.
//
//

#include <metal_stdlib>
using namespace metal;

#define INPUT_TEXTURES_COUNT 10

#define CONCAT(inputAxis, outputAxis, sizeFunction)\
    ushort offset = 0;\
    const auto outputX = gid.x;\
    const auto outputY = gid.y;\
    const auto outputZ = gid.z;\
\
    auto inputX = gid.x;\
    auto inputY = gid.y;\
    auto inputZ = gid.z;\
\
    auto inputTextureIndex = 0;\
\
    for(auto i = 0; i < INPUT_TEXTURES_COUNT; i++) {\
        if (is_null_texture(src[i]))\
            break;\
\
        ushort size = sizeFunction;\
        if (outputAxis < offset + size) {\
            inputTextureIndex = i;\
            inputAxis = outputAxis - offset;\
            break;\
        }\
\
        offset += size;\
    }\
\
    auto input = half4(src[inputTextureIndex].read(ushort2(inputX, inputY), inputZ));\
    dest.write(float4(input), ushort2(outputX, outputY), outputZ);\

/********************** Concat along x **********************/

kernel void concat_x(
                const array<texture2d_array<float, access::read>, INPUT_TEXTURES_COUNT> src [[ texture(0) ]],
                texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputX, outputX, src[i].get_width());

}

kernel void concat_x_3(
                     const array<texture2d<float, access::read>, INPUT_TEXTURES_COUNT> src [[ texture(0) ]],
                     texture2d<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                     ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputX, outputX, src[i].get_width());

}

/********************** Concat along y **********************/

kernel void concat_y(
                   const array<texture2d_array<float, access::read>, INPUT_TEXTURES_COUNT> src [[ texture(0) ]],
                   texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                   ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputY, outputY, src[i].get_height());

}

kernel void concat_y_3(
                    const array<texture2d<float, access::read>, INPUT_TEXTURES_COUNT> src [[ texture(0) ]],
                    texture2d<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                    ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputY, outputY, src[i].get_height());

}

/********************** Concat along z **********************/

kernel void concat_z( /* only works when each input texture has a depth multiple of 4. */
                     const array<texture2d_array<float, access::read>, INPUT_TEXTURES_COUNT> src [[ texture(0) ]],
                     texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                     ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputZ, outputZ, src[i].get_array_size());

}

kernel void concat_z_3(
                    const array<texture2d<float, access::read>, INPUT_TEXTURES_COUNT> src [[ texture(0) ]],
                    texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                    ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputZ, outputZ, 1);

}
