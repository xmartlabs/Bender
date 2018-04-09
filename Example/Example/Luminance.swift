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

    override func validate() {
        let incoming = getIncoming()
        precondition(incoming.count == 2, "Luminance layer must have two inputs")
        precondition(incoming[1].outputSize == incoming[0].outputSize, "Luminance layer must have two inputs with same size")
    }

    override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize
        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let incoming = getIncoming()
        let input1 = incoming[0].getOutput(index: index)
        let input2 = incoming[1].getOutput(index: index)
        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)
        if !enabled {
            rewireIdentity(at: index, image: input1)
            return
        }
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.label = "Luminance encoder"
        encoder.setComputePipelineState(pipelineLuminance)
        encoder.setTexture(input1.texture, index: 0)
        encoder.setTexture(input2.texture, index: 1)
        encoder.setTexture(output.texture, index: 2)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(output.texture.width / threadsPerGroups.width,
                                       output.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        input1.setRead()
        input2.setRead()
    }
}
