//
//  con_transpose.metal
//  Bender
//
//  Created by Mathias Claassen on 3/10/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

#include <metal_stdlib>
#include "bender_metal_types.h"
using namespace metal;

constant ushort filter_x   [[ function_constant(0) ]];
constant ushort filter_y   [[ function_constant(1) ]];

/* ConvTranspose is implemented in three steps. This can most probably be optimized but GPU frame capture says it works faster than Apple's MPSCNNConvolution.
   
 Currently the only supported cases are for textures with more than 4 feature channels (both input and output).
 Another restriction is that kernel sizes and strides must be symmetric and that (stride = kernel_size-1).
 
 Three-step implementation: (kW=kernel_width, kH=kernel_height, iC=input_channels, oC=output_channels, s=stride)
 1- Calculate all multiplications and sums and save them one next to the other in an intermediate texture. Each thread reads its pixel and the corresponding weights (kW*kH*iC*4) and writes (kW*kH*4) pixels to the output.
 2- Shift Left: As the calculations of the neighbor threads should overlap we implement this overlapping by shifting left each block of (kW*kH*oC) to its correct position.
 3- Shift Top: Same as before, shifting top
 */
kernel void transpose_conv_calculate(
                                     texture2d_array<texture_type, access::read> src [[texture(0)]],
                                     texture2d_array<texture_type, access::write> dest [[texture(1)]],

                                     constant float4 *weights [[ buffer(0) ]],

                                     ushort3 gid [[thread_position_in_grid]]
                                     ) {
    // supposes:
    // - weights in HWNC. Should we change this? Tensorflow has HWNC

    // All threads in threadgroup will read the same weights

    ushort in_depth = src.get_array_size();
    ushort output_size = dest.get_array_size() * 4;
    thread calculation_type4 results[9] = {0};

    for (ushort in_z=0; in_z<in_depth; in_z++){
        calculation_type4 in_pixel = calculation_type4(src.read(ushort2(gid.x, gid.y), in_z));

        // loop through filter_x
        for (ushort fx=0; fx<filter_x; fx++)
        {
            // loop through filter_y
            for (ushort fy=0; fy<filter_y; fy++)
            {
                ushort mat_index = fx + filter_x*fy;
                // read weights from 4 kernels
                for (ushort tk=0; tk<4; tk++)
                {
                    // get weights for position
                    calculation_type4 pix = calculation_type4(weights[ushort(in_z +
                                                     ((4 * gid.z + tk) +
                                                      (fy + fx * filter_y) * output_size) * in_depth)]) * in_pixel;

                    results[mat_index][tk] += pix.x + pix.y + pix.z + pix.w;
                }
            }
        }

    }

    // loop through filter_x
    for (ushort fx=0; fx<filter_x; fx++)
    {
        // loop through filter_y
        for (ushort fy=0; fy<filter_y; fy++)
        {
            dest.write(texture_type4(results[fx + filter_y * fy]), ushort2(gid.x * filter_x + fx,
                                                                    gid.y * filter_y + fy), gid.z);
        }
    }
}

kernel void transpose_conv_shift_left(
                                      texture2d_array<texture_type, access::read> src [[texture(0)]],
                                      texture2d_array<texture_type, access::write> dest [[texture(1)]],

                                      ushort3 gid [[thread_position_in_grid]]
                                      ) {

    //base positions: Using x and y switched to avoid different paths
    ushort in_x = filter_x * gid.x;
    ushort out_x = (filter_x - 1) * gid.x;
    ushort in_out_y = filter_y * gid.y;

    // loop through filter_y
    for (ushort fy=0; fy<filter_y; fy++)
    {
        calculation_type4 pix_prev = calculation_type4(src.read(ushort2(in_x - 1, in_out_y + fy), gid.z));
        calculation_type4 pix1 = calculation_type4(src.read(ushort2(in_x, in_out_y + fy), gid.z));
        calculation_type4 pix2 = calculation_type4(src.read(ushort2(in_x + 1, in_out_y + fy), gid.z));
        dest.write(texture_type4(pix_prev + pix1), ushort2(out_x, in_out_y + fy), gid.z);
        dest.write(texture_type4(pix2), ushort2(out_x + 1, in_out_y + fy), gid.z);
    }
}

kernel void transpose_conv_shift_top(
                                      texture2d_array<texture_type, access::read> src [[texture(0)]],
                                      texture2d_array<texture_type, access::write> dest [[texture(1)]],

                                      ushort3 gid [[thread_position_in_grid]]
                                      ) {

    //base positions
    ushort in_out_x = (filter_x - 1) * gid.x;
    ushort in_y = filter_y * gid.y;
    ushort out_y = (filter_y - 1) * gid.y;

    // loop through x
    for (uint fx=0; fx<filter_x-1; fx++)
    {
        calculation_type4 pix_prev = calculation_type4(src.read(ushort2(in_out_x + fx, in_y - 1), gid.z));
        calculation_type4 pix1 = calculation_type4(src.read(ushort2(in_out_x + fx, in_y), gid.z));
        calculation_type4 pix2 = calculation_type4(src.read(ushort2(in_out_x + fx, in_y + 1), gid.z));
        dest.write(texture_type4(pix_prev + pix1), ushort2(in_out_x + fx, out_y), gid.z);
        dest.write(texture_type4(pix2), ushort2(in_out_x + fx, out_y + 1), gid.z);
    }
}
