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

class ConvTransposeLayer: NetworkLayer {

    // fixedBufferSize is maximum size of buffer (count of floats): kHeight * kWidth * inChannels * outChannels = 3*3*4c*2c = 72*c*c
    static let fixedBufferSize = 72 * Constants.baseKernelCount * Constants.baseKernelCount
    var outputLayerSize: LayerSize {
            return size.layerSize
    }

    var descriptor: MPSImageDescriptor?
    let size: ConvSize
    private var prevSize: LayerSize!
    
    let pipelineStateCalculate: MTLComputePipelineState
    let pipelineStateShifLeft: MTLComputePipelineState
    let pipelineStateShiftTop: MTLComputePipelineState
    let neuron: MPSCNNNeuronReLU

    var weightsFiles: [String]?
    var weights: MTLBuffer!
    var inormScaleWeights: MTLBuffer!
    var inormShiftWeights: MTLBuffer!

    var step1Image: MPSImage!
    var step2Image: MPSImage!
    var step3Image: MPSImage!
    var inormOutputImage: MPSImage!
    var outputImage: MPSImage


    init(device: MTLDevice, size: ConvSize, weightsFiles: String...) {
        self.size = size
        self.weightsFiles = weightsFiles
        self.descriptor = MPSImageDescriptor(layerSize: size.layerSize)
        
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

        self.neuron = MPSCNNNeuronReLU(device: device, a: 0)
        self.outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: size.layerSize))
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {

        self.prevSize = prevSize
        let step1ImageSize = LayerSize(f: outputLayerSize.f, w: outputLayerSize.w + prevSize.w)
        let step2ImageSize = LayerSize(f: outputLayerSize.f, w: outputLayerSize.w, h: outputLayerSize.h + prevSize.h)
        self.step1Image = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: step1ImageSize))
        self.step2Image = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: step2ImageSize))
        self.step3Image = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputLayerSize))
        self.inormOutputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputLayerSize))

        if let weightsFiles = weightsFiles {
            let weightVector = loadWeightsForFixedSize(file: weightsFiles[0])
            weights = device.makeBuffer(
                bytes: weightVector,
                length: ConvTransposeLayer.fixedBufferSize * Constants.FloatSize,
                options: MTLResourceOptions.cpuCacheModeWriteCombined)
            inormScaleWeights = device.makeBuffer(
                bytes: loadVectorWeights(fromFilePath: weightsFiles[1], channels: size.layerSize.f),
                length: max(4, size.layerSize.f) * Constants.FloatSize,
                options: [])
            inormShiftWeights = device.makeBuffer(
                bytes: loadVectorWeights(fromFilePath: weightsFiles[2], channels: size.layerSize.f),
                length: max(4, size.layerSize.f) * Constants.FloatSize,
                options: [])
        }
    }

    func updateCheckpoint(new checkpoint: String, old: String, device: MTLDevice) {
        for index in 0..<weightsFiles!.count {
            weightsFiles?[index] = weightsFiles![index].replacingOccurrences(of: old, with: checkpoint, options: String.CompareOptions.anchored)
        }

        if var weightsFiles = weightsFiles {
            let weightVector = loadWeightsForFixedSize(file: weightsFiles[0])
            weights.contents().copyBytes(from: weightVector, count: ConvTransposeLayer.fixedBufferSize * Constants.FloatSize)
            inormScaleWeights.contents().copyBytes(from: loadVectorWeights(fromFilePath: weightsFiles[1], channels: size.layerSize.f), count: max(4, size.layerSize.f) * Constants.FloatSize)
            inormShiftWeights.contents().copyBytes(from: loadVectorWeights(fromFilePath: weightsFiles[2], channels: size.layerSize.f), count: max(4, size.layerSize.f) * Constants.FloatSize)
        }
    }

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {

        // thread group size variables

        let w = pipelineStateCalculate.threadExecutionWidth
        let d = 1
        assert(pipelineStateCalculate.maxTotalThreadsPerThreadgroup / w / d >= 1, "ERROR: wrong thread group size")
        let h = pipelineStateCalculate.maxTotalThreadsPerThreadgroup / w / d


        let threadsPerGroups = MTLSizeMake(w, h, d)
        let threadgroupsPerGrid = MTLSize(width: (inputImage.texture.width + w - 1) / w,
                                          height: (inputImage.texture.height + h - 1) / h,
                                          depth: (outputImage.texture.arrayLength + d - 1) / d)

//        let step1Img = createTempImage(buffer: commandBuffer, descriptor: MPSImageDescriptor(layerSize: step1ImageSize))

        // calculation step
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "convT compute encoder"
        encoder.setComputePipelineState(pipelineStateCalculate)
        encoder.setTexture(inputImage.texture, at: 0)
        encoder.setTexture(step1Image.texture, at: 1)
        encoder.setBuffer(weights, offset: 0, at: 0)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

//        inputImage.setUsed()
//        let step2Img = createTempImage(buffer: commandBuffer, descriptor: MPSImageDescriptor(layerSize: step2ImageSize))

        // shift left step
        let encoder2 = commandBuffer.makeComputeCommandEncoder()
        encoder2.label = "convT shift left encoder"
        encoder2.setComputePipelineState(pipelineStateShifLeft)
        encoder2.setTexture(step1Image.texture, at: 0)
        encoder2.setTexture(step2Image.texture, at: 1)

        encoder2.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder2.endEncoding()

//        step1Img.readCount = 0
//        let step3Img = createTempImage(buffer: commandBuffer)

        // shift top step
        let encoder3 = commandBuffer.makeComputeCommandEncoder()
        encoder3.label = "convT shift top encoder"
        encoder3.setComputePipelineState(pipelineStateShiftTop)
        encoder3.setTexture(step2Image.texture, at: 0)
        encoder3.setTexture(step3Image.texture, at: 1)

        encoder3.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder3.endEncoding()

//        step2Img.readCount = 0
//        let inormOutImg = createTempImage(buffer: commandBuffer)

        ConvolutionLayer.instanceNorm(commandBuffer: commandBuffer,
                                      inputImage: step3Image,
                                      size: self.size.layerSize,
                                      outputImage: inormOutputImage,
                                      scaleBuffer: inormScaleWeights,
                                      shiftBuffer: inormShiftWeights)
//        step3Img.readCount = 0
//        let outputImg = createTempImage(buffer: commandBuffer)

        neuron.encode(commandBuffer: commandBuffer, sourceImage: inormOutputImage, destinationImage: outputImage)
//        inormOutImg.readCount = 0

        return outputImage
    }

    func loadWeightsForFixedSize(file: String) -> UnsafeMutableRawPointer {
        let count = outputLayerSize.f * size.kernelSize * size.kernelSize * prevSize.f
        let bytes = loadConvWeights(fromFilePath: file, prevSize: prevSize, size: size)
        if count == ConvTransposeLayer.fixedBufferSize {
            return UnsafeMutableRawPointer(mutating: bytes)
        } else {
            let vector = UnsafeMutableRawPointer.allocate(bytes: ConvTransposeLayer.fixedBufferSize * Constants.FloatSize,
                                                          alignedTo: MemoryLayout<Float>.alignment)
            vector.copyBytes(from: bytes, count: count * Constants.FloatSize)
            return vector
        }
    }

}
