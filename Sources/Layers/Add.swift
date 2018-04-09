//
//  Add.swift
//  Bender
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Receives two input images and sums them element-wise.
open class Add: NetworkLayer {

    // Custom kernels
    var pipelineAdd: MTLComputePipelineState!

    open override func validate() {
        let incoming = getIncoming()
        // Correctness checks
        assert(incoming.count == 2, "Add only works for two layers (currently \(incoming.count) inputs)")
        assert(incoming[0].outputSize == incoming[1].outputSize, "Add only works for two layers of the same size")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)

        let incoming = getIncoming()
        outputSize = incoming[0].outputSize
        pipelineAdd = MetalShaderManager.shared.getFunction(name: "sum_matrix" + (outputSize.f > 4 ? "" : "_3"), in: Bundle(for: Add.self))

        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let incoming = getIncoming()
        let input1 = incoming[0].getOutput(index: index)
        let input2 = incoming[1].getOutput(index: index)
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder.label = "sum matrix encoder"
        // mean calculation 1st step
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipelineAdd)
        commandEncoder.setTexture(input1.texture, index: 0)
        commandEncoder.setTexture(input2.texture, index: 1)
        commandEncoder.setTexture(getOrCreateOutput(commandBuffer: commandBuffer, index: index).texture, index: 2)
        let threadgroupsPerGrid = input1.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()

        input1.setRead()
        input2.setRead()
    }

}
