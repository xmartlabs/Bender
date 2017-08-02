//
//  LocalResponseNorm.swift
//  Bender
//
//  Created by Diego Ernst on 5/22/17.
//
//

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

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()

        // Correctness checks
        assert(incoming.count == 1, "LocalResponseNorm works for only one image")
        assert(parameters.depthRadius <= 20, "depthRadius must be less or equal to 20")

        outputSize = incoming.first?.outputSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))

        let constants: [FunctionConstantBase] = [
            FunctionConstant(index: 0, type: MTLDataType.ushort, value: parameters.depthRadius),
            FunctionConstant(index: 1, type: MTLDataType.float, value: parameters.bias),
            FunctionConstant(index: 2, type: MTLDataType.float, value: parameters.alpha),
            FunctionConstant(index: 3, type: MTLDataType.float, value: parameters.beta),
        ]
        let isArray = outputSize.f > 4
        pipelineLocalResponseNorm = MetalShaderManager.shared.getFunction(name: "local_response_norm" + (isArray ? "" : "_3"), in: Bundle(for: LocalResponseNorm.self), constants: constants)
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        let incoming = getIncoming()
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.label = "Local Response Norm encoder"
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipelineLocalResponseNorm)

        commandEncoder.setTexture(incoming[0].outputImage.texture, at: 0)
        commandEncoder.setTexture(outputImage.texture, at: 1)
        let threadgroupsPerGrid = incoming[0].outputImage.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()
    }

}
