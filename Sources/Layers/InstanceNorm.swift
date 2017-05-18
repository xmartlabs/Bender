//
//  InstanceNorm.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

open class InstanceNorm: NetworkLayer {

    public static var scaleModifier = "scale"
    public static var shiftModifier = "shift"

    var scale: UnsafePointer<Float>?
    var shift: UnsafePointer<Float>?

    // Intermediate images and buffers
    public var scaleBuffer: MTLBuffer!
    public var shiftBuffer: MTLBuffer!

    var meanPS: MTLComputePipelineState!
    var avgMeanPS: MTLComputePipelineState!
    var varPS: MTLComputePipelineState!
    var avgVarPS: MTLComputePipelineState!
    var inormPS: MTLComputePipelineState!

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "InstanceNorm must have one input, not \(incoming.count)")
        outputSize = incoming[0].outputSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
        scaleBuffer = device.makeBuffer(bytes: scale ?? network.parameterLoader.loadWeights(for: id, modifier: InstanceNorm.scaleModifier, size: outputSize.f),
                                         length: max(4, outputSize.f) * Constants.FloatSize,
                                         options: [])
        shiftBuffer = device.makeBuffer(bytes: shift ?? network.parameterLoader.loadWeights(for: id, modifier: InstanceNorm.shiftModifier, size: outputSize.f),
                                         length: max(4, outputSize.f) * Constants.FloatSize,
                                         options: [])
        let isArray = outputSize.f > 4
        meanPS = MetalShaderManager.shared.getFunction(name: isArray ? "meanA" : "meanA_3", in: Bundle(for: InstanceNorm.self))
        varPS = MetalShaderManager.shared.getFunction(name: isArray ? "varianceA" : "varianceA_3", in: Bundle(for: InstanceNorm.self))
        avgMeanPS = MetalShaderManager.shared.getFunction(name: isArray ? "avgMean" : "avgMean_3", in: Bundle(for: InstanceNorm.self))
        avgVarPS = MetalShaderManager.shared.getFunction(name: isArray ? "avgVar" : "avgVar_3", in: Bundle(for: InstanceNorm.self))
        inormPS = MetalShaderManager.shared.getFunction(name: isArray ? "instanceNorm" : "instanceNorm_3", in: Bundle(for: InstanceNorm.self))
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        guard let network = network else { return }
        scaleBuffer.contents().copyBytes(from: network.parameterLoader.loadWeights(for: id, modifier: InstanceNorm.scaleModifier, size: outputSize.f),
                                          count: max(4, outputSize.f) * Constants.FloatSize)
        shiftBuffer.contents().copyBytes(from: network.parameterLoader.loadWeights(for: id, modifier: InstanceNorm.shiftModifier, size: outputSize.f),
                                          count: max(4, outputSize.f) * Constants.FloatSize)
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {

        let inputImage: MPSImage = getIncoming()[0].outputImage

        let tempImg2 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: outputSize.w, height: 1, featureChannels: outputSize.f))
        let meanImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: outputSize.f))
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.label = "mean encoder"
        // mean calculation 1st step -> Calculate the mean for each column, place on first row
        let tpTG = MTLSizeMake(16, 16, 1)

        commandEncoder.setComputePipelineState(meanPS)

        commandEncoder.setTexture(inputImage.texture, at: 0)
        commandEncoder.setTexture(tempImg2.texture, at: 1)
        let threadgroupsPerGrid = MTLSizeMake(Int(inputImage.texture.height) / tpTG.width, Int((inputImage.texture.arrayLength + tpTG.height - 1) / tpTG.height), 1)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()

        //mean calculation avg (2nd step) -> Calculate the mean of the means in the first row, place result on (0,0)
        let commandEncoder2 = commandBuffer.makeComputeCommandEncoder()
        commandEncoder2.label = "avg mean encoder"
        let threadsPerThreadgroup2 = MTLSizeMake(tempImg2.texture.arrayLength, 1, 1)

        commandEncoder2.setComputePipelineState(avgMeanPS)

        commandEncoder2.setTexture(tempImg2.texture, at: 0)
        commandEncoder2.setTexture(meanImg.texture, at: 1)
        let threadgroupsPerGrid2 = MTLSizeMake(1, 1, 1)
        commandEncoder2.dispatchThreadgroups(threadgroupsPerGrid2, threadsPerThreadgroup: threadsPerThreadgroup2)
        commandEncoder2.endEncoding()

        tempImg2.readCount = 0

        // variance calculation
        let tempImg3 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: outputSize.w, height: 1, featureChannels: outputSize.f))
        let varianceImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: outputSize.f))

        // variance calculation (1st step) -> Variance for each column, results on row 0
        let commandEncoder3 = commandBuffer.makeComputeCommandEncoder()
        commandEncoder3.label = "variance encoder"
        commandEncoder3.setComputePipelineState(varPS)

        commandEncoder3.setTexture(inputImage.texture, at: 0)
        commandEncoder3.setTexture(meanImg.texture, at: 1)
        commandEncoder3.setTexture(tempImg3.texture, at: 2)
        commandEncoder3.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder3.endEncoding()

        // variance calculation avg (2nd step) -> Variance of variances, result on (0,0)
        let commandEncoder4 = commandBuffer.makeComputeCommandEncoder()
        commandEncoder4.label = "avg variance encoder"
        commandEncoder4.setComputePipelineState(avgVarPS)

        commandEncoder4.setTexture(tempImg3.texture, at: 0)
        commandEncoder4.setTexture(varianceImg.texture, at: 1)
        commandEncoder4.dispatchThreadgroups(threadgroupsPerGrid2, threadsPerThreadgroup: threadsPerThreadgroup2)
        commandEncoder4.endEncoding()

        tempImg3.readCount = 0

        // apply instance normalization
        let commandEncoder5 = commandBuffer.makeComputeCommandEncoder()
        commandEncoder5.label = "instance norm encoder"
        commandEncoder5.setComputePipelineState(inormPS)

        commandEncoder5.setTexture(inputImage.texture, at: 0)
        commandEncoder5.setTexture(meanImg.texture, at: 1)
        commandEncoder5.setTexture(varianceImg.texture, at: 2)
        commandEncoder5.setTexture(outputImage.texture, at: 3) // out texture

        commandEncoder5.setBuffer(scaleBuffer, offset: 0, at: 0)
        commandEncoder5.setBuffer(shiftBuffer, offset: 0, at: 1)
        let threadgroupsPerGrid5 = inputImage.texture.threadGrid(threadGroup: tpTG)
        commandEncoder5.dispatchThreadgroups(threadgroupsPerGrid5, threadsPerThreadgroup: tpTG)
        commandEncoder5.endEncoding()

        meanImg.readCount = 0
        varianceImg.readCount = 0
    }
}
