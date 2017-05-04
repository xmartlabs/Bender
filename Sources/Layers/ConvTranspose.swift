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

class ConvTranspose: NetworkLayer {

    var outputSize: LayerSize!

    let size: ConvSize
    private var prevSize: LayerSize!
    
    let pipelineStateCalculate: MTLComputePipelineState
    let pipelineStateShifLeft: MTLComputePipelineState
    let pipelineStateShiftTop: MTLComputePipelineState

    var weightsFile: String
    var weights: MTLBuffer!

    var outputImage: MPSImage


    init(device: MTLDevice, size: ConvSize, neuron: ActivationNeuronType = .relu, weightsFile: String) {
        self.size = size
        self.weightsFile = weightsFile
        
        // Load custom metal kernels

        do {
            let library = device.newDefaultLibrary()!
            let calculateFunc = library.makeFunction(name: "transpose_conv_calculate")
            let shiftLeftFunc = library.makeFunction(name: "transpose_conv_shift_left")
            let shiftTopFunc = library.makeFunction(name: "transpose_conv_shift_top")
            calculateFunc?.label = "convT_calculate"
            shiftLeftFunc?.label = "convT Shift Left"
            shiftTopFunc?.label = "convT Shift Top"
            pipelineStateCalculate = try device.makeComputePipelineState(function: calculateFunc!)
            pipelineStateShifLeft = try device.makeComputePipelineState(function: shiftLeftFunc!)
            pipelineStateShiftTop = try device.makeComputePipelineState(function: shiftTopFunc!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }

        self.outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {

        self.prevSize = prevSize

        updateWeights(device: device)
    }

    func updateCheckpoint(new checkpoint: String, old: String, device: MTLDevice) {
        weightsFile = weightsFile.replacingOccurrences(of: old, with: checkpoint, options: String.CompareOptions.anchored)

        updateWeights(device: device)
    }

    func updateWeights(device: MTLDevice) {
        let vector = loadWeights(from: weightsFile, size: getWeightsSize())
        weights.contents().copyBytes(from: vector, count: getWeightsSize())
    }

    func getWeightsSize() -> Int {
        return prevSize.f * size.kernelSize * size.kernelSize * size.outputChannels
    }

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage {

        // thread group size variables

        let w = pipelineStateCalculate.threadExecutionWidth
        let d = 1
        assert(pipelineStateCalculate.maxTotalThreadsPerThreadgroup / w / d >= 1, "ERROR: wrong thread group size")
        let h = pipelineStateCalculate.maxTotalThreadsPerThreadgroup / w / d

        let step1ImageSize = LayerSize(f: outputSize.f, w: outputSize.w + prevSize.w)
        let step2ImageSize = LayerSize(f: outputSize.f, w: outputSize.w, h: outputSize.h + prevSize.h)

        let threadsPerGroups = MTLSizeMake(w, h, d)
        let threadgroupsPerGrid = MTLSize(width: (inputImage.texture.width + w - 1) / w,
                                          height: (inputImage.texture.height + h - 1) / h,
                                          depth: (outputImage.texture.arrayLength + d - 1) / d)

        let step1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step1ImageSize))

        // calculation step
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "convT compute encoder"
        encoder.setComputePipelineState(pipelineStateCalculate)
        encoder.setTexture(inputImage.texture, at: 0)
        encoder.setTexture(step1Img.texture, at: 1)
        encoder.setBuffer(weights, offset: 0, at: 0)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        let step2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step2ImageSize))

        // shift left step
        let encoder2 = commandBuffer.makeComputeCommandEncoder()
        encoder2.label = "convT shift left encoder"
        encoder2.setComputePipelineState(pipelineStateShifLeft)
        encoder2.setTexture(step1Img.texture, at: 0)
        encoder2.setTexture(step2Img.texture, at: 1)

        encoder2.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder2.endEncoding()

        step1Img.readCount = 0

        // shift top step
        let encoder3 = commandBuffer.makeComputeCommandEncoder()
        encoder3.label = "convT shift top encoder"
        encoder3.setComputePipelineState(pipelineStateShiftTop)
        encoder3.setTexture(step2Img.texture, at: 0)
        encoder3.setTexture(outputImage.texture, at: 1)

        encoder3.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder3.endEncoding()

        step2Img.readCount = 0

        return outputImage
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
