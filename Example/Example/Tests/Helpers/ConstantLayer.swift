//
//  ConstantLayer.swift
//  Example
//
//  Created by Diego Ernst on 6/28/17.
//
//

import MetalBender
import MetalPerformanceShaders
import MetalPerformanceShadersProxy

open class Constant: NetworkLayer {

    let constantOutputTexture: Texture
    var constantOutputImage: MPSImage!

    init(outputTexture: Texture) {
        constantOutputTexture = outputTexture
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming: [NetworkLayer] = getIncoming()
        assert(incoming.count == 1, "Constant must have one input, not \(incoming.count)")
        outputSize = constantOutputTexture.size
        constantOutputImage = MPSImage(texture: constantOutputTexture.metalTexture(with: device), featureChannels: constantOutputTexture.size.f)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        rewireIdentity(at: executionIndex, image: constantOutputImage)
    }

}
