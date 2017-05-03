//
//  LuminanceLayer.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 4/25/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

class LuminanceLayer: NetworkLayer {

    var outputLayerSize: LayerSize {
        return layerSize
    }

    var layerSize: LayerSize!

    // Custom kernels
    let pipelineLuminance: MTLComputePipelineState

    // Intermediate images
    var descriptor: MPSImageDescriptor?
    var outputImage: MPSImage!

    init(device: MTLDevice, layerSize: LayerSize) {
        self.layerSize = layerSize
        do {
            let library = device.newDefaultLibrary()!
            let kernel = library.makeFunction(name: "luminance_transfer")
            pipelineLuminance = try device.makeComputePipelineState(function: kernel!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        descriptor = MPSImageDescriptor(channelFormat: .float16, width: layerSize.w, height: layerSize.h, featureChannels: layerSize.f)
        outputImage = createImage(device: device)
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {
        guard let originalImage = originalImage else {
            return inputImage
        }

        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "Luminance encoder"
        encoder.setComputePipelineState(pipelineLuminance)
        encoder.setTexture(inputImage.texture, at: 0)
        encoder.setTexture(originalImage.texture, at: 1)
        encoder.setTexture(outputImage.texture, at: 2)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        return outputImage
    }
}
