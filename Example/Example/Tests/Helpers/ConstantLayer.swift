//
//  ConstantLayer.swift
//  Example
//
//  Created by Diego Ernst on 6/28/17.
//
//

import Bender
import MetalPerformanceShadersProxy

open class Constant: NetworkLayer {

    let constantOutputTexture: Texture
    var constantOutputImage: MPSImage!

    init(outputTexture: Texture) {
        constantOutputTexture = outputTexture
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "Constant must have one input, not \(incoming.count)")
        outputSize = constantOutputTexture.size
        constantOutputImage = MPSImage(texture: constantOutputTexture.metalTexture(with: device), featureChannels: constantOutputTexture.size.f)
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        outputImage = constantOutputImage
    }

}
