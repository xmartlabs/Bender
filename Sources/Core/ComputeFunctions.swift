//
//  ComputeFunctions.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import Foundation

class ComputeFunctions {

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
    static var scaleToFloatPS: MTLComputePipelineState!

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
            let scaleToFloat = defaultLibrary.makeFunction(name: "scale_to_float")

            ComputeFunctions.meanPS = try device.makeComputePipelineState(function: meanKernel!)
            ComputeFunctions.avgMeanPS =  try device.makeComputePipelineState(function: avgMeanKernel!)
            ComputeFunctions.avgVarPS =  try device.makeComputePipelineState(function: avgVarKernel!)
            ComputeFunctions.variancePS =  try device.makeComputePipelineState(function: varianceKernel!)
            ComputeFunctions.inormPS =  try device.makeComputePipelineState(function: inormKernel!)
            ComputeFunctions.meanPS_3 = try device.makeComputePipelineState(function: meanKernel_3!)
            ComputeFunctions.avgMeanPS_3 =  try device.makeComputePipelineState(function: avgMeanKernel_3!)
            ComputeFunctions.avgVarPS_3 =  try device.makeComputePipelineState(function: avgVarKernel_3!)
            ComputeFunctions.variancePS_3 =  try device.makeComputePipelineState(function: varianceKernel_3!)
            ComputeFunctions.inormPS_3 =  try device.makeComputePipelineState(function: inormKernel_3!)
            ComputeFunctions.sumMatrixPS =  try device.makeComputePipelineState(function: sumMatrixKernel!)
            ComputeFunctions.scaleToFloatPS = try device.makeComputePipelineState(function: scaleToFloat!)

        } catch {
            fatalError("Unable to create pipeline state, check metal shaders")
        }
    }
    
}
