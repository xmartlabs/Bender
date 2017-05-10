//
//  BGRAtoRGBA.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShaders

open class BGRAtoRGBA: NetworkLayer {

    // Custom kernels
    let pipelineBGRAtoRGBA: MTLComputePipelineState!

    public init(device: MTLDevice, id: String? = nil) {
        // Load custom metal kernels
        pipelineBGRAtoRGBA = MetalShaderManager.shared.getFunction(name: "bgra_to_rgba", in: Bundle(for: BGRAtoRGBA.self))
        super.init(id: id)
    }

    open override func initialize(device: MTLDevice) {
        outputSize = getIncoming().first?.outputSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    open override func execute(commandBuffer: MTLCommandBuffer) {
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "BGRA to RGBA encoder"
        encoder.setComputePipelineState(pipelineBGRAtoRGBA)
        encoder.setTexture(getIncoming()[0].outputImage.texture, at: 0)
        encoder.setTexture(outputImage.texture, at: 1)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()
    }
    
}
