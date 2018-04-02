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

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize

        kernel = MPSCNNSoftMax(device: device)
        createOutputs(size: outputSize)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        kernel.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputs[executionIndex], destinationImage: outputs[executionIndex])
    }
}
