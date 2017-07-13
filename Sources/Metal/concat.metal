//
//  concat.metal
//  Bender
//
//  Created by Diego Ernst on 6/27/17.
//
//

#include <metal_stdlib>
using namespace metal;

/********************** Concat MARCO **********************/

#define INPUT_TEXTURES_COUNT (10)

#define INPUT_TEXTURE(i) src ## _ ## i

#define COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, i)\
    {\
        if (is_null_texture(INPUT_TEXTURE(i)))\
            return;\
        \
        ushort size = INPUT_TEXTURE(i).sizeFunction;\
        if (outputAxis < offset + size) {\
            inputAxis = outputAxis - offset;\
            auto input = half4(INPUT_TEXTURE(i).read(ushort2(inputX, inputY), inputZ));\
            dest.write(float4(input), ushort2(outputX, outputY), outputZ);\
            return;\
        }\
        \
        offset += size;\
    }

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
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 0);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 1);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 2);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 3);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 4);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 5);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 6);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 7);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 8);\
    COMPUTE_INPUT_TEXTURE(inputAxis, outputAxis, sizeFunction, 9);\

/********** Helpers to define a list of input textures *******/

#define TEXTURE(type, name, i) type name ## _ ## i [[ texture(i) ]]

#define TEXTURES_ARRAY(type, name)\
    TEXTURE(type, name, 0),\
    TEXTURE(type, name, 1),\
    TEXTURE(type, name, 2),\
    TEXTURE(type, name, 3),\
    TEXTURE(type, name, 4),\
    TEXTURE(type, name, 5),\
    TEXTURE(type, name, 6),\
    TEXTURE(type, name, 7),\
    TEXTURE(type, name, 8),\
    TEXTURE(type, name, 9)\

/*** typedefs in order to avois issues with ',' in macros ***/

typedef const texture2d_array<float, access::read> texture2d_float_array;
typedef const texture2d<float, access::read> texture2d_float;

/********************** Concat along x **********************/

kernel void concat_x(
                TEXTURES_ARRAY(texture2d_float_array, src),
                texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputX, outputX, get_width());

}

kernel void concat_x_3(
                     TEXTURES_ARRAY(texture2d_float, src),
                     texture2d<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                     ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputX, outputX, get_width());

}

/********************** Concat along y **********************/

kernel void concat_y(
                   TEXTURES_ARRAY(texture2d_float_array, src),
                   texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                   ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputY, outputY, get_height());

}

kernel void concat_y_3(
                    TEXTURES_ARRAY(texture2d_float, src),
                    texture2d<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                    ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputY, outputY, get_height());

}

/********************** Concat along z **********************/

kernel void concat_z( /* only works when each input texture has a depth multiple of 4. */
                     TEXTURES_ARRAY(texture2d_float_array, src),
                     texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                     ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputZ, outputZ, get_array_size());

}

kernel void concat_z_3(
                    TEXTURES_ARRAY(texture2d_float, src),
                    texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                    ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputZ, outputZ, get_width() & 0x00 | 0x01); /* Just want to pass 1 as the sizeFunction */

}
