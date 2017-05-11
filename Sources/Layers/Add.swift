//
//  Add.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders

/// Receives two input images. The first is used to take the color and the second is used to take the luminance for the output image.
open class Add: NetworkLayer {

    // Custom kernels
    let pipelineAdd: MTLComputePipelineState

    public init(device: MTLDevice, id: String? = nil) {
        pipelineAdd = MetalShaderManager.shared.getFunction(name: "sum_matrix", in: Bundle(for: Add.self))
        super.init(id: id)
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        //TODO: check that all prevSizes are of the same size
        let incoming = getIncoming()
        assert(incoming.count == 2, "Add works for two layers")
        outputSize = incoming.first?.outputSize
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
