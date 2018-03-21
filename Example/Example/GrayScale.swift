//
//  GrayScale.swift
//  Example
//
//  Created by Mathias Claassen on 5/30/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy
import MetalBender

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

    override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        assert(getIncoming().count == 1, "GrayScale must have one input")
        let incoming = getIncoming()[0]
        assert(incoming.outputSize.f <= 4, "GrayScale input must have at most 4 feature channels")
        outputSize = LayerSize(h: incoming.outputSize.h, w: incoming.outputSize.w, f: outputChannels)
        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let incoming = getIncoming()[0]
        let input = incoming.getOutput(index: index)
        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.label = "GrayScale encoder"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(input.texture, index: 0)
        encoder.setTexture(output.texture, index: 1)
        let threadsPerGroups = MTLSizeMake(32, 4, 1)
        let threadGroups = output.texture.threadGrid(threadGroup: threadsPerGroups)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        input.setRead()
    }
}
