//
//  LuminanceLayer.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 4/25/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import Palladium

/// Receives two input images. The first is used to take the color and the second is used to take the luminance for the output image.
class Luminance: NetworkLayer {

    // Custom kernels
    let pipelineLuminance: MTLComputePipelineState

    init(device: MTLDevice, id: String? = nil) {
        pipelineLuminance = MetalShaderManager.shared.getFunction(name: "luminance_transfer")
        super.init(id: id)
    }

    override func initialize(device: MTLDevice) {
        //TODO: check that all prevSizes are of the same size
        outputSize = getIncoming().first?.outputSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    override func execute(commandBuffer: MTLCommandBuffer) {
        let incoming = getIncoming()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "Luminance encoder"
        encoder.setComputePipelineState(pipelineLuminance)
        encoder.setTexture(incoming[0].outputImage.texture, at: 0)
        encoder.setTexture(incoming[1].outputImage.texture, at: 1)
        encoder.setTexture(outputImage.texture, at: 2)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
    }
}
