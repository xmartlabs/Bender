//
//  con_transpose.metal
//  VideoStylizer
//
//  Created by Mathias Claassen on 3/10/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

constant ushort filter_x   [[ function_constant(0) ]];
constant ushort filter_y   [[ function_constant(1) ]];

kernel void transpose_conv_calculate(
                                     texture2d_array<float, access::read> src [[texture(0)]],
                                     texture2d_array<float, access::write> dest [[texture(1)]],

                                     constant float4 *weights [[ buffer(0) ]],

                                     ushort3 gid [[thread_position_in_grid]]
                                     ) {
    // supposes:
    // - weights in HWNC. Should we change this? Tensorflow has HWNC

    // All threads in threadgroup will read the same weights

    ushort in_depth = src.get_array_size();
    ushort output_size = dest.get_array_size() * 4;
    thread half4 results[9] = {0};

    for (ushort in_z=0; in_z<in_depth; in_z++){
        half4 in_pixel = half4(src.read(ushort2(gid.x, gid.y), in_z));

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
                    half4 pix = half4(weights[ushort(in_z + in_depth *
                                                     ((4 * gid.z + tk) + output_size *
                                                      (fx + filter_x * (fy))))]) * in_pixel;

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
            dest.write(float4(results[fx + filter_x * fy]), ushort2(gid.x * filter_x + fx,
                                                                    gid.y * filter_y + fy), gid.z);
        }
    }
}

kernel void transpose_conv_shift_left(
                                      texture2d_array<float, access::read> src [[texture(0)]],
                                      texture2d_array<float, access::write> dest [[texture(1)]],

                                      ushort3 gid [[thread_position_in_grid]]
                                      ) {

    //base positions: Using x and y switched to avoid different paths
    ushort in_x = filter_x * gid.x;
    ushort out_x = (filter_x - 1) * gid.x;
    ushort in_out_y = filter_y * gid.y;

    // loop through filter_y
    for (ushort fy=0; fy<filter_y; fy++)
    {
        half4 pix_prev = half4(src.read(ushort2(in_x - 1, in_out_y + fy), gid.z));
        half4 pix1 = half4(src.read(ushort2(in_x, in_out_y + fy), gid.z));
        half4 pix2 = half4(src.read(ushort2(in_x + 1, in_out_y + fy), gid.z));
        dest.write(float4(pix_prev + pix1), ushort2(out_x, in_out_y + fy), gid.z);
        dest.write(float4(pix2), ushort2(out_x + 1, in_out_y + fy), gid.z);
    }
}

kernel void transpose_conv_shift_top(
                                      texture2d_array<float, access::read> src [[texture(0)]],
                                      texture2d_array<float, access::write> dest [[texture(1)]],

                                      ushort3 gid [[thread_position_in_grid]]
                                      ) {

    //base positions
    ushort in_out_x = (filter_x - 1) * gid.x;
    ushort in_y = filter_y * gid.y;
    ushort out_y = (filter_y - 1) * gid.y;

    // loop through x
    for (uint fx=0; fx<filter_x-1; fx++)
    {
        half4 pix_prev = half4(src.read(ushort2(in_out_x + fx, in_y - 1), gid.z));
        half4 pix1 = half4(src.read(ushort2(in_out_x + fx, in_y), gid.z));
        half4 pix2 = half4(src.read(ushort2(in_out_x + fx, in_y + 1), gid.z));
        dest.write(float4(pix_prev + pix1), ushort2(in_out_x + fx, out_y), gid.z);
        dest.write(float4(pix2), ushort2(in_out_x + fx, out_y + 1), gid.z);
    }
}
