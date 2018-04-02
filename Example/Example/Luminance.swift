//
//  LuminanceLayer.swift
//  Bender
//
//  Created by Mathias Claassen on 4/25/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy
import MetalBender

/// Receives two input images. The first is used to take the luminance and the second is used to take the color for the output image. Used for color preservation
class Luminance: NetworkLayer {

    var enabled: Bool

    // Custom kernels
    let pipelineLuminance: MTLComputePipelineState

    init(enabled: Bool, id: String? = nil) {
        self.enabled = enabled
        pipelineLuminance = MetalShaderManager.shared.getFunction(name: "luminance_transfer")
        super.init(id: id)
    }

    override func initialize(network: Network, device: MTLDevice) {
        let incoming = getIncoming()

        precondition(incoming.count == 2, "Luminance layer must have two inputs")
        precondition(incoming[1].outputSize == incoming[0].outputSize, "Luminance layer must have two inputs with same size")
        super.initialize(network: network, device: device)
        outputSize = incoming[0].outputSize
        createOutputs(size: outputSize)
    }

    override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        let incoming = getIncoming()
        if !enabled {
            outputs[executionIndex] = incoming[0].outputs[executionIndex]
            return
        }
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.label = "Luminance encoder"
        encoder.setComputePipelineState(pipelineLuminance)
        encoder.setTexture(incoming[0].outputs[executionIndex].texture, index: 0)
        encoder.setTexture(incoming[1].outputs[executionIndex].texture, index: 1)
        encoder.setTexture(outputs[executionIndex].texture, index: 2)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputs[executionIndex].texture.width / threadsPerGroups.width,
                                       outputs[executionIndex].texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
    }
}
