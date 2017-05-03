//
//  ScaleToFloatLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

/**
 Performs a scale to the matrix and then adjusts [0-255] -> [0.0-1.0]
 **/
class ScaleToFloatLayer: NetworkLayer {
    var descriptor: MPSImageDescriptor?

    var outputLayerSize = LayerSize(f: 3, w: 256)
    
    // Custom kernels
    let pipelineScaleToFloat: MTLComputePipelineState
    
    // Intermediate images
    let outputImage : MPSImage
    
    init(device: MTLDevice) {
        // Load custom metal kernels
        do {
            let library = device.newDefaultLibrary()!
            let kernel = library.makeFunction(name: "scale_to_float")
            pipelineScaleToFloat = try device.makeComputePipelineState(function: kernel!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }
        
        // init intermediate images
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputLayerSize))
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {}

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "ScaleToFloat encoder"
        encoder.setComputePipelineState(pipelineScaleToFloat)
        encoder.setTexture(inputImage.texture, at: 0)
        encoder.setTexture(outputImage.texture, at: 1)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
        
        return outputImage
    }
    
    func onFinish(){
        print("input to Scale Layer")
//        Test.printTexturePixel(inputImage!.texture)
    }
}
