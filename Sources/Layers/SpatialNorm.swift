//
//  SpatialNormalization.swift
//  Bender
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShadersProxy

/// Implements Spatial normalization.
open class SpatialNorm: NetworkLayer {

    public var kernel: MPSCNNSpatialNormalization!
    var kWidth: Int
    var kHeight: Int
    var alpha: Float?
    var beta: Float?
    var delta: Float?


    public init(kWidth: Int, kHeight: Int, alpha: Float? = nil, beta: Float? = nil, delta: Float? = nil, id: String? = nil) {
        assert(alpha == nil || alpha! >= 0.0, "Alpha must be non-negative")
        self.kWidth = kWidth
        self.kHeight = kHeight
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        super.init(id: id)
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "SpatialNorm must have one input, not \(incoming.count)")
        outputSize = incoming[0].outputSize

        kernel = MPSCNNSpatialNormalization(device: device, kernelWidth: kWidth, kernelHeight: kHeight)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))

        if let alpha = alpha { kernel.alpha = alpha }
        if let beta = beta { kernel.beta = beta }
        if let delta = delta { kernel.delta = delta }
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        kernel.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputImage, destinationImage: outputImage)
    }
}
