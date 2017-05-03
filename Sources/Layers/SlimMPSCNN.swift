/*
	Copyright (C) 2016 Apple Inc. All Rights Reserved.
	See LICENSE.txt for this sample’s licensing information
	
	Abstract:
	This file describes slimmer routines to create some common MPSCNNFunctions, it is useful especially to fetch network parameters from .dat files
 */

import Foundation
import MetalPerformanceShaders

class SlimMPSCNNConvolution: MPSCNNConvolution{
    
    /**
         Encode a MPSCNNKernel into a command Buffer. The operation shall proceed out-of-place.
         
         We calculate the appropriate offset as per how TensorFlow calculates its padding using input image size and stride here.
         
         This [Link](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py) has an explanation in header comments how tensorFlow pads its convolution input images.
         
         - Parameters:
             - commandBuffer: A valid MTLCommandBuffer to receive the encoded filter
             - sourceImage: A valid MPSImage object containing the source image.
             - destinationImage: A valid MPSImage to be overwritten by result image. destinationImage may not alias sourceImage
     */
    func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage, padding: Bool) {
        
        // select offset according to padding being used or not
        if(padding){
            let pad_along_height = ((destinationImage.height - 1) * strideInPixelsY + kernelHeight - sourceImage.height)
            let pad_along_width  = ((destinationImage.width - 1) * strideInPixelsX + kernelWidth - sourceImage.width)
            let pad_top = Int(pad_along_height / 2)
            let pad_left = Int(pad_along_width / 2)
            
            self.offset = MPSOffset(x: ((Int(kernelWidth)/2) - pad_left), y: (Int(kernelHeight/2) - pad_top), z: 0)
        }
        else{
            self.offset = MPSOffset(x: Int(kernelWidth)/2, y: Int(kernelHeight)/2, z: 0)
        }
        super.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
    
}
