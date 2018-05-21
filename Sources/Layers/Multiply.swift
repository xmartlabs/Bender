//
//  Multiply.swift
//  MetalBender
//
//  Created by Mathias Claassen on 5/21/18.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Receives one input image and multiplies it by a scalar.
open class Multiply: NetworkLayer {

    // Custom kernels
    var pipelineMul: MTLComputePipelineState!
    var scalar: Float

    public init(scalar: Float, id: String?) {
        self.scalar = scalar
        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()
        // Correctness checks
        assert(incoming.count == 1, "Multiply only works for one layer, (currently \(incoming.count) inputs)")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)

        let incoming = getIncoming()
        outputSize = incoming[0].outputSize
        pipelineMul = MetalShaderManager.shared.getFunction(name: "multiply_scalar" + (outputSize.f > 4 ? "" : "_3"),
                                                            in: Bundle(for: Add.self),
                                                            constants: [FunctionConstant(index: 0, type: .float, value: scalar)])

        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let input1 = getIncoming()[0].getOutput(index: index)
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder.label = "sum matrix encoder"
        // mean calculation 1st step
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipelineMul)
        commandEncoder.setTexture(input1.texture, index: 0)
        commandEncoder.setTexture(getOrCreateOutput(commandBuffer: commandBuffer, index: index).texture, index: 2)
        let threadgroupsPerGrid = input1.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()

        input1.setRead()
    }

}
