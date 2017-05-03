//
//  ResidualBlockLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

class ResidualLayer: NetworkLayer {
    var descriptor: MPSImageDescriptor?
    
    let id: String
    let size: LayerSize
    var outputLayerSize: LayerSize {
        get { return size }
    }
    let weightsFiles: [String]
    var convA: ConvolutionLayer!
    var convB: ConvolutionLayer!
    var outputImage: MPSImage!
    
    init(size: LayerSize, weightFiles: String...) {
        self.size = size
        self.weightsFiles = weightFiles
        self.id = UUID().uuidString
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {
        let convSize = ConvSize(layerSize: size, kernelSize: 3, stride: 1)
        convA = ConvolutionLayer(size: convSize, neuronType: .relu, weightsFiles: weightsFiles[0], weightsFiles[1], weightsFiles[2])
        convA?.initialize(device: device, prevSize: prevSize)
        convB = ConvolutionLayer(size: convSize, neuronType: .none, weightsFiles: weightsFiles[3], weightsFiles[4], weightsFiles[5])
        convB?.initialize(device: device, prevSize: size)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: size))
    }

    func updateCheckpoint(new checkpoint: String, old: String, device: MTLDevice) {
        convA.updateCheckpoint(new: checkpoint, old: old, device: device)
        convB.updateCheckpoint(new: checkpoint, old: old, device: device)
    }
    
    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {
        
        let out1 = convA.execute(commandBuffer: commandBuffer, inputImage: inputImage, originalImage: originalImage)
        let out2 = convB.execute(commandBuffer: commandBuffer, inputImage: out1, originalImage: originalImage)
        
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.label = "sum matrix encoder"
        // mean calculation 1st step
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(NetworkPipeline.sumMatrixPS)
        
        commandEncoder.setTexture(inputImage.texture, at: 0)
        commandEncoder.setTexture(out2.texture, at: 1)
        commandEncoder.setTexture(outputImage.texture, at: 2)
        let threadgroupsPerGrid = inputImage.texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()
        
        return outputImage
        
    }
    
}

