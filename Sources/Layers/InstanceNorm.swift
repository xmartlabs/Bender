//
//  InstanceNorm.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

class InstanceNorm: NetworkLayer {

    var outputSize: LayerSize!

    // Intermediate images and buffers
    var scaleFilename: String
    var shiftFilename: String
    var scaleWeights: MTLBuffer!
    var shiftWeights: MTLBuffer!
    var outputImage: MPSImage!

    init(scaleFile: String, shiftFile: String) {
        self.scaleFilename = scaleFile
        self.shiftFilename = shiftFile
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        outputSize = prevSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: prevSize))
        scaleWeights = device.makeBuffer(bytes: loadVectorWeights(fromFilePath: scaleFilename, channels: prevSize.f),
                                         length: max(4, prevSize.f) * Constants.FloatSize,
                                         options: [])
        shiftWeights = device.makeBuffer(bytes: loadVectorWeights(fromFilePath: shiftFilename, channels: prevSize.f),
                                         length: max(4, prevSize.f) * Constants.FloatSize,
                                         options: [])
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {
        scaleFilename = scaleFilename.replacingOccurrences(of: old, with: new, options: String.CompareOptions.anchored)
        shiftFilename = shiftFilename.replacingOccurrences(of: old, with: new, options: String.CompareOptions.anchored)

        scaleWeights.contents().copyBytes(from: loadVectorWeights(fromFilePath: scaleFilename, channels: outputSize.f), count: max(4, outputSize.f) * Constants.FloatSize)
        shiftWeights.contents().copyBytes(from: loadVectorWeights(fromFilePath: shiftFilename, channels: outputSize.f), count: max(4, outputSize.f) * Constants.FloatSize)
    }

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage {

        var meanPS, avgMeanPS, varPS, avgVarPS, inormPS: MTLComputePipelineState
        let isArray = outputSize.f > 4
        meanPS = isArray ? ComputeFunctions.meanPS : ComputeFunctions.meanPS_3
        avgMeanPS = isArray ? ComputeFunctions.avgMeanPS : ComputeFunctions.avgMeanPS_3
        varPS = isArray ? ComputeFunctions.variancePS : ComputeFunctions.variancePS_3
        avgVarPS = isArray ? ComputeFunctions.avgVarPS : ComputeFunctions.avgVarPS_3
        inormPS = isArray ? ComputeFunctions.inormPS : ComputeFunctions.inormPS_3

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

        commandEncoder5.setBuffer(scaleWeights, offset: 0, at: 0)
        commandEncoder5.setBuffer(shiftWeights, offset: 0, at: 1)
        let threadgroupsPerGrid5 = inputImage.texture.threadGrid(threadGroup: tpTG)
        commandEncoder5.dispatchThreadgroups(threadgroupsPerGrid5, threadsPerThreadgroup: tpTG)
        commandEncoder5.endEncoding()

        meanImg.readCount = 0
        varianceImg.readCount = 0

        return outputImage
    }
}
