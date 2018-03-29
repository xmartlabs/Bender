//
//  InstanceNorm.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Instance normalization layer
open class InstanceNorm: NetworkLayer {

    public static var scaleModifier = "scale"
    public static var shiftModifier = "shift"

    var scale: Data?
    var shift: Data?

    // Intermediate images and buffers
    public var scaleBuffer: MTLBuffer!
    public var shiftBuffer: MTLBuffer!

    var inormPS: MTLComputePipelineState!

    public init(scale: Data? = nil, shift: Data? = nil, id: String? = nil) {
        self.scale = scale
        self.shift = shift
        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "InstanceNorm must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize
        createOutputs(size: outputSize, temporary: temporaryImage)
        scaleBuffer = device.makeBuffer(bytes: scale?.pointer() ?? network.parameterLoader.loadWeights(for: id,
                                                                                                       modifier: InstanceNorm.scaleModifier,
                                                                                                       size: outputSize.f),
                                         length: outputSize.f * Constants.FloatSize,
                                         options: [])
        shiftBuffer = device.makeBuffer(bytes: shift?.pointer() ?? network.parameterLoader.loadWeights(for: id,
                                                                                                       modifier: InstanceNorm.shiftModifier,
                                                                                                       size: outputSize.f),
                                         length: outputSize.f * Constants.FloatSize,
                                         options: [])
        let isArray = outputSize.f > 4
        inormPS = MetalShaderManager.shared.getFunction(name: isArray ? "instance_norm" : "instance_norm_3", in: Bundle(for: InstanceNorm.self))
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        guard let network = network else { return }
        scaleBuffer.contents().copyBytes(from: network.parameterLoader.loadWeights(for: id, modifier: InstanceNorm.scaleModifier, size: outputSize.f),
                                          count: max(4, outputSize.f) * Constants.FloatSize)
        shiftBuffer.contents().copyBytes(from: network.parameterLoader.loadWeights(for: id, modifier: InstanceNorm.shiftModifier, size: outputSize.f),
                                          count: max(4, outputSize.f) * Constants.FloatSize)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {

        let inputImage: MPSImage = getIncoming()[0].getOutput(index: index)
        let maxThreads = 256
        let threadWidth = min(maxThreads, outputSize.w)
        let threadHeight = min(maxThreads / threadWidth, outputSize.h)
        let tpTG = MTLSizeMake(threadWidth, threadHeight, 1)

        // apply instance normalization
        let commandEncoder5 = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder5.label = "instance norm encoder"
        commandEncoder5.setComputePipelineState(inormPS)

        commandEncoder5.setTexture(inputImage.texture, index: 0)
        commandEncoder5.setTexture(getOrCreateOutput(commandBuffer: commandBuffer, index: index).texture, index: 1) // out texture

        commandEncoder5.setBuffer(scaleBuffer, offset: 0, index: 0)
        commandEncoder5.setBuffer(shiftBuffer, offset: 0, index: 1)
        commandEncoder5.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: inputImage.texture.arrayLength), threadsPerThreadgroup: tpTG)
        commandEncoder5.endEncoding()

        inputImage.setRead()
    }

}
