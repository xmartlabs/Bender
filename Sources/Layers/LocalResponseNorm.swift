//
//  LocalResponseNorm.swift
//  Bender
//
//  Created by Diego Ernst on 5/22/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Implements Local Response Normalization (LRN).
open class LocalResponseNorm: NetworkLayer {

    public struct Parameters {

        public let depthRadius: Int
        public let bias: Float
        public let alpha: Float
        public let beta: Float

        public init(depthRadius: Int = 5, bias: Float = 1, alpha: Float = 1, beta: Float = 1) {
            self.depthRadius = depthRadius
            self.bias = bias
            self.alpha = alpha
            self.beta = beta
        }

    }

    var pipelineLocalResponseNorm: MTLComputePipelineState!

    let parameters: Parameters

    public init(parameters: Parameters = Parameters(), id: String? = nil) {
        self.parameters = parameters
        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()

        // Correctness checks
        assert(incoming.count == 1, "LocalResponseNorm works for only one image")
        assert(parameters.depthRadius <= 20, "depthRadius must be less or equal to 20")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()

        outputSize = incoming.first?.outputSize
        createOutputs(size: outputSize, temporary: temporaryImage)

        let constants: [FunctionConstantBase] = [
            FunctionConstant(index: 0, type: MTLDataType.ushort, value: parameters.depthRadius),
            FunctionConstant(index: 1, type: MTLDataType.float, value: parameters.bias),
            FunctionConstant(index: 2, type: MTLDataType.float, value: parameters.alpha),
            FunctionConstant(index: 3, type: MTLDataType.float, value: parameters.beta)
        ]
        let isArray = outputSize.f > 4
        pipelineLocalResponseNorm = MetalShaderManager.shared.getFunction(name: "local_response_norm" + (isArray ? "" : "_3"),
                                                                          in: Bundle(for: LocalResponseNorm.self),
                                                                          constants: constants)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let incoming = getIncoming()[0]
        let input = incoming.getOutput(index: index)
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder.label = "Local Response Norm encoder"
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipelineLocalResponseNorm)

        commandEncoder.setTexture(input.texture, index: 0)
        commandEncoder.setTexture(getOrCreateOutput(commandBuffer: commandBuffer, index: index).texture, index: 1)
        let threadgroupsPerGrid = input.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()
    }

}
