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

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "Identity must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        rewireIdentity(at: executionIndex, image: getIncoming()[0].getOutput(index: executionIndex))
    }

}
