//
//  SoftMax.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShaders

open class SoftMax: NetworkLayer {

    public var kernel: MPSCNNSoftMax!

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        outputSize = getIncoming()[0].outputSize

        kernel = MPSCNNSoftMax(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        kernel.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputImage, destinationImage: outputImage)
    }
}

