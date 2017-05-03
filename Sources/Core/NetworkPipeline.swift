//
//  NetworkPipeline.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 11/25/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

struct NetworkPipeline {
    var layers: [NetworkLayer]
    var device: MTLDevice
    static var meanPS: MTLComputePipelineState!
    static var avgMeanPS: MTLComputePipelineState!
    static var avgVarPS: MTLComputePipelineState!
    static var variancePS: MTLComputePipelineState!
    static var inormPS: MTLComputePipelineState!
    static var meanPS_3: MTLComputePipelineState!
    static var avgMeanPS_3: MTLComputePipelineState!
    static var avgVarPS_3: MTLComputePipelineState!
    static var variancePS_3: MTLComputePipelineState!
    static var inormPS_3: MTLComputePipelineState!
    static var sumMatrixPS: MTLComputePipelineState!

    var preserveColors = true
    
    init(device: MTLDevice, inputSize: LayerSize, layers: [NetworkLayer]) {
        self.layers = layers
        self.device = device
        
        setupPipelineStates(device: device)
        
        initialize(inputSize: inputSize)
    }
    
    mutating private func initialize(inputSize: LayerSize) {
        var prevSize = inputSize
        for layer in layers {
            layer.initialize(device: device, prevSize: prevSize)
            prevSize = layer.outputLayerSize
        }
    }
    
    func run(inputImage: MPSImage, commandBuffer: MTLCommandBuffer, result: @escaping (MPSImage) -> Void) {
        autoreleasepool {
            let croppedImage = layers[0].execute(commandBuffer: commandBuffer, inputImage: inputImage, originalImage: inputImage)
            var image = croppedImage
            for layer in layers[1..<layers.count] {
                let layerOutput = layer.execute(commandBuffer: commandBuffer,
                                                inputImage: image,
                                                originalImage: preserveColors ? croppedImage : nil)
                image = layerOutput
            }
            commandBuffer.commit()
            //TODO: We should execute this on another queue
            commandBuffer.waitUntilCompleted()
            result(image)
        }
    }
    
    private func setupPipelineStates(device: MTLDevice) {
        do {
            let defaultLibrary = device.newDefaultLibrary()!
            
            let meanKernel =  defaultLibrary.makeFunction(name: "meanA")
            meanKernel?.label = "meanKernel"
            let varianceKernel =  defaultLibrary.makeFunction(name: "varianceA")
            varianceKernel?.label = "varianceKernel"
            let avgMeanKernel =  defaultLibrary.makeFunction(name: "avgMean")
            let avgVarKernel =  defaultLibrary.makeFunction(name: "avgVar")
            let inormKernel =  defaultLibrary.makeFunction(name: "instanceNorm")
            let meanKernel_3 =  defaultLibrary.makeFunction(name: "meanA_3")
            let varianceKernel_3 =  defaultLibrary.makeFunction(name: "varianceA_3")
            let avgMeanKernel_3 =  defaultLibrary.makeFunction(name: "avgMean_3")
            let avgVarKernel_3 =  defaultLibrary.makeFunction(name: "avgVar_3")
            let inormKernel_3 =  defaultLibrary.makeFunction(name: "instanceNorm_3")
            let sumMatrixKernel =  defaultLibrary.makeFunction(name: "sum_matrix")

            NetworkPipeline.meanPS = try device.makeComputePipelineState(function: meanKernel!)
            NetworkPipeline.avgMeanPS =  try device.makeComputePipelineState(function: avgMeanKernel!)
            NetworkPipeline.avgVarPS =  try device.makeComputePipelineState(function: avgVarKernel!)
            NetworkPipeline.variancePS =  try device.makeComputePipelineState(function: varianceKernel!)
            NetworkPipeline.inormPS =  try device.makeComputePipelineState(function: inormKernel!)
            NetworkPipeline.meanPS_3 = try device.makeComputePipelineState(function: meanKernel_3!)
            NetworkPipeline.avgMeanPS_3 =  try device.makeComputePipelineState(function: avgMeanKernel_3!)
            NetworkPipeline.avgVarPS_3 =  try device.makeComputePipelineState(function: avgVarKernel_3!)
            NetworkPipeline.variancePS_3 =  try device.makeComputePipelineState(function: varianceKernel_3!)
            NetworkPipeline.inormPS_3 =  try device.makeComputePipelineState(function: inormKernel_3!)
            NetworkPipeline.sumMatrixPS =  try device.makeComputePipelineState(function: sumMatrixKernel!)
        }
        catch
        {
            fatalError("Unable to create pipeline state, check metal shaders")
        }
    }
}
