//
//  Add.swift
//  Bender
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShadersProxy

/// Receives two input images and sums them element-wise.
open class Add: NetworkLayer {

    // Custom kernels
    var pipelineAdd: MTLComputePipelineState!

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()

        // Correctness checks
        assert(incoming.count == 2, "Add works for two layers")
        assert(incoming[0].outputSize == incoming[1].outputSize, "Add works for two layers of the same size")

        outputSize = incoming[0].outputSize
        pipelineAdd = MetalShaderManager.shared.getFunction(name: "sum_matrix" + (outputSize.f > 4 ? "" : "_3"), in: Bundle(for: Add.self))

        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        let incoming = getIncoming()
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.label = "sum matrix encoder"
        // mean calculation 1st step
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipelineAdd)

        commandEncoder.setTexture(incoming[0].outputImage.texture, at: 0)
        commandEncoder.setTexture(incoming[1].outputImage.texture, at: 1)
        commandEncoder.setTexture(outputImage.texture, at: 2)
        let threadgroupsPerGrid = incoming[0].outputImage.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()
    }
    
}
