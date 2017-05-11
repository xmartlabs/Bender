//
//  ScaleToImage.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShaders

open class ScaleToImage: NetworkLayer {

    // Custom kernels
    let pipeline: MTLComputePipelineState!

    public init(device: MTLDevice, scale: Float = 0.5, shift: Float = 0.5, id: String? = nil) {
        // Load custom metal kernels
        pipeline = MetalShaderManager.shared.getFunction(name: "scale_to_image",
                                                         in: Bundle(for: ScaleToImage.self),
                                                         constants: [FunctionConstant<Float>(index: 0, type: MTLDataType.float, value: scale),
                                                                     FunctionConstant<Float>(index: 1, type: MTLDataType.float, value: shift)])
        super.init(id: id)
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        outputSize = getIncoming().first?.outputSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "Scale to Image encoder"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(getIncoming()[0].outputImage.texture, at: 0)
        encoder.setTexture(outputImage.texture, at: 1)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
    }
    
}
