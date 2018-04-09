//
//  Softmax.swift
//  Bender
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Applies Softmax
open class Softmax: NetworkLayer {

    public var kernel: MPSCNNSoftMax!

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "SoftMax must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize

        kernel = MPSCNNSoftMax(device: device)
        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        kernel.encode(commandBuffer: commandBuffer,
                      sourceImage: getIncoming()[0].getOutput(index: index),
                      destinationImage: getOrCreateOutput(commandBuffer: commandBuffer, index: index))
    }
}
