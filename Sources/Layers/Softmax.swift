//
//  Softmax.swift
//  Bender
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShadersProxy

/// Applies Softmax
open class Softmax: NetworkLayer {

    public var kernel: MPSCNNSoftMax!

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "SoftMax must have one input, not \(incoming.count)")
        outputSize = getIncoming()[0].outputSize

        kernel = MPSCNNSoftMax(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        kernel.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputImage, destinationImage: outputImage)
    }
}

