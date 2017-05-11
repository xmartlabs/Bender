//
//  LanczosLayer.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 4/24/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

open class Scale: NetworkLayer {

    public var lanczos: MPSImageLanczosScale!

    public init(layerSize: LayerSize, id: String? = nil) {
        super.init(id: id)
        self.outputSize = layerSize
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        lanczos = MPSImageLanczosScale(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: getIncoming()[0].outputImage.texture, destinationTexture: outputImage.texture)
    }
}
