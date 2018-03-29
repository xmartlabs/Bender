//
//  Concat.swift
//  Bender
//
//  Created by Diego Ernst on 6/27/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

open class Concat: NetworkLayer {

    var pipeline: MTLComputePipelineState!
    let axis: LayerSizeAxis

    let maxInputTextures = 10 // must match kernel max input textures

    public init(axis: LayerSizeAxis, id: String? = nil) {
        self.axis = axis
        super.init(id: id)
    }

    open override func validate() {
        let axisThatMustBeEqual = LayerSizeAxis.all.filter { $0 != axis }
        validatePreconditions(with: axisThatMustBeEqual)
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let axisThatMustBeEqual = LayerSizeAxis.all.filter { $0 != axis }
        let incoming = getIncoming()

        var outputDimensions = [LayerSizeAxis: Int]()
        outputDimensions[axis] = incoming.reduce(0) { result, networkLayer in result + networkLayer.outputSize[axis] }
        axisThatMustBeEqual.forEach { outputDimensions[$0] = incoming[0].outputSize[$0] }

        outputSize = LayerSize(h: outputDimensions[.h]!, w: outputDimensions[.w]!, f: outputDimensions[.f]!)
        createOutputs(size: outputSize, temporary: temporaryImage)

        var shaderFunc = ""
        switch axis {
        case .h:
            shaderFunc = "concat_y"
        case .w:
            shaderFunc = "concat_x"
        case .f:
            shaderFunc = "concat_z"
        }
        if incoming[0].outputSize.f <= 4 {
            shaderFunc += "_3"
        }
        pipeline = MetalShaderManager.shared.getFunction(name: shaderFunc, in: Bundle(for: Concat.self))
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        let incoming = getIncoming()
        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)

        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        commandEncoder.label = "Concat encoder"
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipeline)

        (0..<min(maxInputTextures, incoming.count)).forEach {
            commandEncoder.setTexture(incoming[$0].getOutput(index: index).texture, index: $0)
        }

        commandEncoder.setTexture(output.texture, index: maxInputTextures)
        let threadgroupsPerGrid = output.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()

        incoming.forEach {
            $0.getOutput(index: index).setRead()
        }
    }

    private func validatePreconditions(with axisThatMustBeEqual: [LayerSizeAxis]) {
        let incoming = getIncoming()

        // TODO: implement shader to support concat along z without restricting it to multiples of 4

        assert(!incoming.isEmpty, "Concat: expects at least one inputs")
        assert(axis != .f || incoming.reduce(true) { result, networkLayer in result && (networkLayer.outputSize.f % 4 == 0) },
               "Concat: all z dimensions must be multiple of 4")
        assert(incoming.count <= maxInputTextures, "Concat: only accepts \(maxInputTextures) incomming nodes at most")

        let allInputTexturesWithMoreThan4Channels = incoming.reduce(true) { result, networkLayer in result && networkLayer.outputSize.f > 4 }
        let allInputTexturesWithLessThanOrEqualTo4Channels = incoming.reduce(true) { result, networkLayer in
            result && networkLayer.outputSize.f <= 4
        }
        assert(allInputTexturesWithMoreThan4Channels || allInputTexturesWithLessThanOrEqualTo4Channels,
               "All z dimensions must be either > 4 or <= 4 at the same time")

        var axisValues = [LayerSizeAxis: Int]()

        let sampleSize: LayerSize! = incoming[0].outputSize
        axisThatMustBeEqual.forEach { axisValues[$0] = sampleSize[$0] }

        incoming.forEach {
            let inputSize: LayerSize! = $0.outputSize
            axisThatMustBeEqual.forEach {
                assert(inputSize[$0] == axisValues[$0]!, "Concat: Axis \($0) isn't equal in at least one pair of input nodes")
            }
        }
    }

}
