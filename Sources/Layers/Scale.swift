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

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        lanczos = MPSImageLanczosScale(device: device)
        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let input = getIncoming()[0].getOutput(index: index)
        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)
        lanczos.encode(commandBuffer: commandBuffer,
                       sourceTexture: input.texture,
                       destinationTexture: output.texture)
        input.setRead()
    }
}
