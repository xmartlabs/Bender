//
//  ImageLinearTransform.swift
//  Bender
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Performs a scaling and clamps the output values to between 0 and 1
open class ImageLinearTransform: NetworkLayer {

    // Custom kernels
    var pipeline: MTLComputePipelineState!
    var shift: Float
    var scale: Float

    public init(scale: Float = 0.5, shift: Float = 0.5, id: String? = nil) {
        self.scale = scale
        self.shift = shift
        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "ImageLinearTransform must have one input, not \(incoming.count)")
        assert(incoming[0].outputSize.f == 3 || incoming[0].outputSize.f == 4,
               "ImageLinearTransform should only be used if it has 3 or 4 feature channels as input")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize
        // Load custom metal kernels
        pipeline = MetalShaderManager.shared.getFunction(name: "image_linear_transform",
                                                         in: Bundle(for: ImageLinearTransform.self),
                                                         constants: [FunctionConstant<Float>(index: 0, type: MTLDataType.float, value: scale),
                                                                     FunctionConstant<Float>(index: 1, type: MTLDataType.float, value: shift)])

        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let input = getIncoming()[0].getOutput(index: index)
        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.label = "Scale to Image encoder"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(input.texture, index: 0)
        encoder.setTexture(output.texture, index: 1)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = output.texture.threadGrid(threadGroup: threadsPerGroups)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        input.setRead()
    }

}
