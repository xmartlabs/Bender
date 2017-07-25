//
//  Identity.swift
//  Bender
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShadersProxy

/// Identity layer. Returns the input image
open class Identity: NetworkLayer {

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "Identity must have one input, not \(incoming.count)")
        outputSize = incoming[0].outputSize
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        outputImage = getIncoming()[0].outputImage
    }

}
