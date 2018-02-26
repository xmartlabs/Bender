//
//  BatchNorm.swift
//  MetalBender
//
//  Created by Mathias Claassen on 2/12/18.
//

import MetalPerformanceShaders

/// Implements Batch normalization.
open class BatchNorm: NetworkLayer {

    public static var meanModifier = "mean"
    public static var varianceModifier = "variance"
    public static var scaleModifier = "scale"
    public static var offsetModifier = "offset"

    var kernel: MTLComputePipelineState!
    var params: Data?
    var allParamsSet = false
    var epsilon: Float

    public var paramBuffer: MTLBuffer!

    public init(mean: Data? = nil, variance: Data? = nil, offset: Data? = nil, scale: Data? = nil, epsilon: Float = 0, id: String? = nil) {
        self.epsilon = epsilon
        if let mean = mean, let variance = variance {
            params = mean
            params?.append(variance)

            if let scale = scale {
                params?.append(scale)
            } else {
                let scaleArr = [Float](repeating: 1.0, count: variance.count / 4)
                params?.append(UnsafeBufferPointer(start: scaleArr, count: scaleArr.count))
            }

            if let offset = offset {
                params?.append(offset)
            } else {
                let offsetArr = [Float](repeating: 0.0, count: variance.count / 4)
                params?.append(UnsafeBufferPointer(start: offsetArr, count: offsetArr.count))
            }
            params?.append(UnsafeBufferPointer(start: [epsilon], count: 1))
            params = params?.toFloat16()
            allParamsSet = true

        } else if let scale = scale, let offset = offset {
            params = scale
            params?.append(offset)
        }

        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "BatchNorm must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize

        let isArray = outputSize.f > 4
        kernel = MetalShaderManager.shared.getFunction(name: "batch_norm" + (isArray ? "" : "_3"),
                                                       in: Bundle(for: BatchNorm.self))
        if !allParamsSet {
            var d1 = Data(bytes: network.parameterLoader.loadWeights(for: id, modifier: BatchNorm.meanModifier, size: outputSize.f),
                          count: outputSize.f * Constants.FloatSize)
            d1.append(Data(bytes: network.parameterLoader.loadWeights(for: id, modifier: BatchNorm.varianceModifier, size: outputSize.f),
                           count: outputSize.f * Constants.FloatSize))

            if let params = self.params {
                d1.append(params)
            } else {
                d1.append(Data(bytes: network.parameterLoader.loadWeights(for: id, modifier: BatchNorm.scaleModifier, size: outputSize.f),
                               count: outputSize.f * Constants.FloatSize))
                d1.append(Data(bytes: network.parameterLoader.loadWeights(for: id, modifier: BatchNorm.offsetModifier, size: outputSize.f),
                               count: outputSize.f * Constants.FloatSize))
            }
            d1.append(UnsafeBufferPointer(start: [epsilon], count: 1))
            self.params = d1.toFloat16()
        }
        if let params = params {
            paramBuffer = device.makeBuffer(bytes: params.pointer()!,
                                            length: (max(4, outputSize.f) * 4 + 1) * Constants.HalfSize,
                                            options: [])
        }
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.label = "Batch Norm encoder"
        encoder.setComputePipelineState(kernel)
        encoder.setTexture(getIncoming()[0].outputImage.texture, index: 0)
        encoder.setTexture(outputImage.texture, index: 1)

        encoder.setBuffer(paramBuffer, offset: 0, index: 0)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = outputImage.texture.threadGrid(threadGroup: threadsPerGroups)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
    }
}
