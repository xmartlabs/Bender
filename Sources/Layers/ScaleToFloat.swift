//
//  ScaleToFloatLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

/**
 Adjusts [-1.0-1.0] -> [0.0-1.0]
 **/
class ScaleToFloat: NetworkLayer {

    var outputSize: LayerSize!

    // Intermediate images
    var outputImage : MPSImage!
    
    init() {}
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {
        outputSize = prevSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: prevSize))
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage {
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "ScaleToFloat encoder"
        encoder.setComputePipelineState(ComputeFunctions.scaleToFloatPS)
        encoder.setTexture(inputImage.texture, at: 0)
        encoder.setTexture(outputImage.texture, at: 1)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
        
        return outputImage
    }
    
}
