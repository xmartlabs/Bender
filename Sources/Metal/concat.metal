//
//  concat.metal
//  Bender
//
//  Created by Diego Ernst on 6/27/17.
//
//

#include <metal_stdlib>
using namespace metal;

/********************** Concat MACRO **********************/

/* Changing this value implies changing the
 'SRC_TEXTURES_ARRAY' & 'CONCAT' macros as well.
 Additionally it must match the property 'maxInputTextures'
 of the Concat class defined in the Concat.swift file.
 */
#define INPUT_TEXTURES_COUNT (10)

#define INPUT_TEXTURE(i) src_ ## i

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

#define TEXTURE(type, i) type INPUT_TEXTURE(i) [[ texture(i) ]]

#define SRC_TEXTURES_ARRAY(type)\
    TEXTURE(type, 0),\
    TEXTURE(type, 1),\
    TEXTURE(type, 2),\
    TEXTURE(type, 3),\
    TEXTURE(type, 4),\
    TEXTURE(type, 5),\
    TEXTURE(type, 6),\
    TEXTURE(type, 7),\
    TEXTURE(type, 8),\
    TEXTURE(type, 9)\

/*** typedefs in order to avoid issues with ',' in macros ***/

typedef const texture2d_array<float, access::read> texture2d_float_array;
typedef const texture2d<float, access::read> texture2d_float;

/********************** Concat along x **********************/

kernel void concat_x(
                SRC_TEXTURES_ARRAY(texture2d_float_array),
                texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputX, outputX, get_width());

}

kernel void concat_x_3(
                     SRC_TEXTURES_ARRAY(texture2d_float),
                     texture2d<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                     ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputX, outputX, get_width());

}

/********************** Concat along y **********************/

kernel void concat_y(
                   SRC_TEXTURES_ARRAY(texture2d_float_array),
                   texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                   ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputY, outputY, get_height());

}

kernel void concat_y_3(
                    SRC_TEXTURES_ARRAY(texture2d_float),
                    texture2d<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                    ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputY, outputY, get_height());

}

/********************** Concat along z **********************/

kernel void concat_z( /* only works when each input texture has a depth multiple of 4. */
                     SRC_TEXTURES_ARRAY(texture2d_float_array),
                     texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                     ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputZ, outputZ, get_array_size());

}

kernel void concat_z_3(
                    SRC_TEXTURES_ARRAY(texture2d_float),
                    texture2d_array<float, access::write> dest [[ texture(INPUT_TEXTURES_COUNT) ]],
                    ushort3 gid [[thread_position_in_grid]]) {

    CONCAT(inputZ, outputZ, get_width() * 0 + 1); /* Just want to pass 1 as the sizeFunction */

}
