//
//  Concat.swift
//  Bender
//
//  Created by Diego Ernst on 6/27/17.
//
//

import MetalPerformanceShadersProxy

open class Concat: NetworkLayer {

    var pipeline: MTLComputePipelineState!
    let axis: LayerSizeAxis

    let maxInputTextures = 10 // must match kernel max input textures

    public init(axis: LayerSizeAxis, id: String? = nil) {
        self.axis = axis
        super.init(id: id)
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)

        let axisThatMustBeEqual = LayerSizeAxis.all.filter { $0 != axis }
        let (allInputTexturesWithLessThanOrEqualTo4Channels, axisValues) = validatePreconditions(with: axisThatMustBeEqual)

        let incoming = getIncoming()
        var outputDimensions = [LayerSizeAxis: Int]()
        outputDimensions[axis] = incoming.reduce(0) { $0.0 + $0.1.outputSize[axis] }
        axisThatMustBeEqual.forEach { outputDimensions[$0] = axisValues[$0] }

        outputSize = LayerSize(f: outputDimensions[.f]!, w: outputDimensions[.w]!, h: outputDimensions[.h]!)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))

        var shaderFunc = ""
        switch axis {
        case .h:
            shaderFunc = "concat_y"
        case .w:
            shaderFunc = "concat_x"
        case .f:
            shaderFunc = "concat_z"
        }
        if allInputTexturesWithLessThanOrEqualTo4Channels {
            shaderFunc += "_3"
        }
        pipeline = MetalShaderManager.shared.getFunction(name: shaderFunc, in: Bundle(for: Concat.self))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        let incoming = getIncoming()
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.label = "Concat encoder"
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipeline)

        (0..<min(maxInputTextures, incoming.count)).forEach {
            commandEncoder.setTexture(incoming[$0].outputImage.texture, at: $0)
        }

        commandEncoder.setTexture(outputImage.texture, at: maxInputTextures)
        let threadgroupsPerGrid = outputImage.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()
    }

    private func validatePreconditions(with axisThatMustBeEqual: [LayerSizeAxis]) -> (Bool, [LayerSizeAxis: Int]) {
        let incoming = getIncoming()

        // TODO: implement shader to support concat along z without restricting it to multiples of 4

        assert(!incoming.isEmpty, "Concat: expects at least one inputs")
        assert(axis != .f || incoming.reduce(true) { $0.0 && ($0.1.outputSize.f % 4 == 0) }, "Concat: all z dimensions must be multiple of 4")
        assert(incoming.count <= maxInputTextures, "Concat: only accepts \(maxInputTextures) incomming nodes at most")

        let allInputTexturesWithMoreThan4Channels = incoming.reduce(true) { $0.0 && $0.1.outputSize.f > 4 }
        let allInputTexturesWithLessThanOrEqualTo4Channels = incoming.reduce(true) { $0.0 && $0.1.outputSize.f <= 4 }
        assert(allInputTexturesWithMoreThan4Channels || allInputTexturesWithLessThanOrEqualTo4Channels, "All z dimensions must be either > 4 or <= 4 at the same time")

        var axisValues = [LayerSizeAxis: Int]()

        let sampleSize: LayerSize! = incoming[0].outputSize
        axisThatMustBeEqual.forEach { axisValues[$0] = sampleSize[$0] }

        incoming.forEach {
            let inputSize: LayerSize! = $0.outputSize
            axisThatMustBeEqual.forEach {
                assert(inputSize[$0] == axisValues[$0]!, "Concat: Axis \($0) isn't equal in at least one pair of input nodes")
            }
        }
        return (allInputTexturesWithLessThanOrEqualTo4Channels, axisValues)
    }

}
