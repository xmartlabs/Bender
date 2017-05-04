//
//  LuminanceLayer.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 4/25/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

/// Receives two input images. The first is used to take the color and the second is used to take the luminance for the output image.
class Luminance: NetworkLayerUnion {

    var outputSize: LayerSize!

    // Custom kernels
    let pipelineLuminance: MTLComputePipelineState

    // Intermediate images
    var outputImage: MPSImage!

    init(device: MTLDevice) {
        do {
            let library = device.newDefaultLibrary()!
            let kernel = library.makeFunction(name: "luminance_transfer")
            pipelineLuminance = try device.makeComputePipelineState(function: kernel!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        outputSize = prevSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: prevSize))
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImages: MPSImage...) -> MPSImage {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "Luminance encoder"
        encoder.setComputePipelineState(pipelineLuminance)
        encoder.setTexture(inputImages[0].texture, at: 0)
        encoder.setTexture(inputImages[1].texture, at: 1)
        encoder.setTexture(outputImage.texture, at: 2)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        return outputImage
    }
}
