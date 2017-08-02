//
//  GrayScale.swift
//  Example
//
//  Created by Mathias Claassen on 5/30/17.
//
//

import MetalPerformanceShadersProxy
import Bender

/// Receives two input images. The first is used to take the color and the second is used to take the luminance for the output image.
class GrayScale: NetworkLayer {

    // Custom kernels
    let pipeline: MTLComputePipelineState
    let outputChannels: Int

    init(outputChannels: Int = 1, id: String? = nil) {
        pipeline = MetalShaderManager.shared.getFunction(name: "to_grayscale")
        self.outputChannels = outputChannels
        super.init(id: id)
    }

    override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        assert(getIncoming().count == 1, "GrayScale must have one input")
        let incoming = getIncoming()[0]
        assert(incoming.outputSize.f <= 4, "GrayScale input must have at most 4 feature channels")
        outputSize = LayerSize(f: outputChannels, w: incoming.outputSize.w, h: incoming.outputSize.h)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    override func execute(commandBuffer: MTLCommandBuffer) {
        let incoming = getIncoming()
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "GrayScale encoder"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(incoming[0].outputImage.texture, at: 0)
        encoder.setTexture(outputImage.texture, at: 1)
        let threadsPerGroups = MTLSizeMake(32, 4, 1)
        let threadGroups = outputImage.texture.threadGrid(threadGroup: threadsPerGroups)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
    }
}
