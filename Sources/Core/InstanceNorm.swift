//
//  InstanceNorm.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

extension NetworkLayer {
    
    static func instanceNorm(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, size: LayerSize, outputImage: MPSImage, scaleBuffer: MTLBuffer, shiftBuffer: MTLBuffer) {
        var meanPS, avgMeanPS, varPS, avgVarPS, inormPS: MTLComputePipelineState
        let isArray = size.f > 4
        meanPS = isArray ? NetworkPipeline.meanPS : NetworkPipeline.meanPS_3
        avgMeanPS = isArray ? NetworkPipeline.avgMeanPS : NetworkPipeline.avgMeanPS_3
        varPS = isArray ? NetworkPipeline.variancePS : NetworkPipeline.variancePS_3
        avgVarPS = isArray ? NetworkPipeline.avgVarPS : NetworkPipeline.avgVarPS_3
        inormPS = isArray ? NetworkPipeline.inormPS : NetworkPipeline.inormPS_3
        
        let tempImg2 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: size.w, height: 1, featureChannels: size.f))
        let meanImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: size.f))
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
        let tempImg3 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: size.w, height: 1, featureChannels: size.f))
        let varianceImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: size.f))
        
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
    
    func loadVectorWeights(fromFilePath filePath: String, channels: Int) -> UnsafePointer<Float> {
        return loadWeights(fromFilePath: filePath, size: channels, useFloat16: false)
    }
    
    func loadConvWeights(fromFilePath filePath: String, prevSize: LayerSize, size: ConvSize, useFloat16: Bool = false) -> UnsafePointer<Float> {
        return loadWeights(fromFilePath: filePath, size: prevSize.f * size.kernelSize * size.kernelSize * size.layerSize.f, useFloat16: useFloat16)
    }
    
    private func loadWeights(fromFilePath filePath: String, size: Int, useFloat16: Bool) -> UnsafePointer<Float> {
        // Load weights from file(s)
        let typeSize = useFloat16 ? Constants.HalfSize : Constants.FloatSize
        let sizeWeights = size * typeSize
        
        // get the url to this layer's weights
        let wtPath = Bundle.main.path( forResource: filePath, ofType: "")
        
        // open file descriptors in read-only mode to parameter files
        let fd_w  = open( wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)

        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")
        
        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0);
        
        // cast Void pointers to Float
        let w = UnsafePointer(hdrW!.bindMemory(to: Float.self, capacity: Int(sizeWeights)))

        close(fd_w)
        assert(w != UnsafePointer<Float>.init(bitPattern: -1), "mmap failed with errno = \(errno)")
        return w
    }
    
}

