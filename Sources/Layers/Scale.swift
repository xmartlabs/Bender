//
//  LanczosLayer.swift
//  Bender
//
//  Created by Mathias Claassen on 4/24/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Scales the input image to a given size
open class Scale: NetworkLayer {

    public var lanczos: MPSImageLanczosScale!

    public init(layerSize: LayerSize, id: String? = nil) {
        super.init(id: id)
        self.outputSize = layerSize
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "Scale must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        lanczos = MPSImageLanczosScale(device: device)
        createOutputs(size: outputSize)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: getIncoming()[0].outputs[executionIndex].texture, destinationTexture: outputs[executionIndex].texture)
    }
}
