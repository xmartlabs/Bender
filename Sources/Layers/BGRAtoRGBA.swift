//
//  BGRAtoRGBA.swift
//  Bender
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Transforms an image from RGBA to BGRA. (You can use it the other way around too)
open class BGRAtoRGBA: NetworkLayer {

    // Custom kernels
    let pipelineBGRAtoRGBA: MTLComputePipelineState!

    public override init(id: String? = nil) {
        // Load custom metal kernels
        pipelineBGRAtoRGBA = MetalShaderManager.shared.getFunction(name: "bgra_to_rgba", in: Bundle(for: BGRAtoRGBA.self))
        super.init(id: id)
    }
    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "BGRAtoRGBA supports one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize
        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let input = getIncoming()[0].getOutput(index: index)
        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)

        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.label = "BGRA to RGBA encoder"
        encoder.setComputePipelineState(pipelineBGRAtoRGBA)
        encoder.setTexture(input.texture, index: 0)
        encoder.setTexture(output.texture, index: 1)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = output.texture.threadGrid(threadGroup: threadsPerGroups)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        input.setRead()
    }

}
