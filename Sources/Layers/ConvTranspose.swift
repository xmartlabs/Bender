//
//  ConvTransposeZerosLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

struct WeightData {
    var count: UInt32
    var data: UnsafePointer<Float>
}

open class ConvTranspose: NetworkLayer {

    static var weightModifier: String = ""
    
    let size: ConvSize
    private var prevSize: LayerSize!
    
    let pipelineCalculate: MTLComputePipelineState
    let pipelineShifLeft: MTLComputePipelineState
    let pipelineShiftTop: MTLComputePipelineState

    var weights: MTLBuffer!

    public init(device: MTLDevice, size: ConvSize, neuron: ActivationNeuronType = .relu, id: String? = nil) {
        self.size = size
        // Load custom metal kernels
        pipelineCalculate = MetalShaderManager.shared.getFunction(name: "transpose_conv_calculate", in: Bundle(for: ConvTranspose.self))
        pipelineShifLeft = MetalShaderManager.shared.getFunction(name: "transpose_conv_shift_left", in: Bundle(for: ConvTranspose.self))
        pipelineShiftTop = MetalShaderManager.shared.getFunction(name: "transpose_conv_shift_top", in: Bundle(for: ConvTranspose.self))
        super.init(id: id)
    }
    
    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "ConvTranspose must have one input, not \(incoming.count)")
        prevSize = incoming[0].outputSize
        outputSize = LayerSize(f: size.outputChannels,
                                    w: prevSize.w * size.stride)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))

        weights = device.makeBuffer(bytes: network.parameterLoader.loadWeights(for: id, modifier: ConvTranspose.weightModifier, size: getWeightsSize()),
                                    length: getWeightsSize() * Constants.FloatSize,
                                    options: [])
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        guard let network = network else { return }
        let vector = network.parameterLoader.loadWeights(for: id, modifier: ConvTranspose.weightModifier, size: getWeightsSize())
        weights.contents().copyBytes(from: vector, count: getWeightsSize())
    }

    open func getWeightsSize() -> Int {
        return prevSize.f * size.kernelSize * size.kernelSize * size.outputChannels
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {

        // thread group size variables
        let incoming = getIncoming()
        let w = pipelineCalculate.threadExecutionWidth
        let d = 1
        assert(pipelineCalculate.maxTotalThreadsPerThreadgroup / w / d >= 1, "ERROR: wrong thread group size")
        let h = pipelineCalculate.maxTotalThreadsPerThreadgroup / w / d

        let step1ImageSize = LayerSize(f: outputSize.f, w: outputSize.w + prevSize.w)
        let step2ImageSize = LayerSize(f: outputSize.f, w: outputSize.w, h: outputSize.h + prevSize.h)

        let threadsPerGroups = MTLSizeMake(w, h, d)
        let threadgroupsPerGrid = MTLSize(width: (incoming[0].outputImage.texture.width + w - 1) / w,
                                          height: (incoming[0].outputImage.texture.height + h - 1) / h,
                                          depth: (outputImage.texture.arrayLength + d - 1) / d)

        let step1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step1ImageSize))

        // calculation step
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "convT compute encoder"
        encoder.setComputePipelineState(pipelineCalculate)
        encoder.setTexture(incoming[0].outputImage.texture, at: 0)
        encoder.setTexture(step1Img.texture, at: 1)
        encoder.setBuffer(weights, offset: 0, at: 0)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        let step2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step2ImageSize))

        // shift left step
        let encoder2 = commandBuffer.makeComputeCommandEncoder()
        encoder2.label = "convT shift left encoder"
        encoder2.setComputePipelineState(pipelineShifLeft)
        encoder2.setTexture(step1Img.texture, at: 0)
        encoder2.setTexture(step2Img.texture, at: 1)

        encoder2.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder2.endEncoding()

        step1Img.readCount = 0

        // shift top step
        let encoder3 = commandBuffer.makeComputeCommandEncoder()
        encoder3.label = "convT shift top encoder"
        encoder3.setComputePipelineState(pipelineShiftTop)
        encoder3.setTexture(step2Img.texture, at: 0)
        encoder3.setTexture(outputImage.texture, at: 1)

        encoder3.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder3.endEncoding()

        step2Img.readCount = 0
    }

//    func loadWeightsForFixedSize(file: String) -> UnsafeMutableRawPointer {
//        let count = outputLayerSize.f * size.kernelSize * size.kernelSize * prevSize.f
//        let bytes = loadConvWeights(fromFilePath: file, prevSize: prevSize, size: size)
//        if count == ConvTransposeLayer.fixedBufferSize {
//            return UnsafeMutableRawPointer(mutating: bytes)
//        } else {
//            let vector = UnsafeMutableRawPointer.allocate(bytes: ConvTransposeLayer.fixedBufferSize * Constants.FloatSize,
//                                                          alignedTo: MemoryLayout<Float>.alignment)
//            vector.copyBytes(from: bytes, count: count * Constants.FloatSize)
//            return vector
//        }
//    }

}
