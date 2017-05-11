//
//  ResidualBlockLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

/** Residual layer. Receives 6 or 8 weight files:
    [convA_weights, instance_norm_A_scale, instance_norm_A_shift, convB_weights, instance_norm_B_scale, instance_norm_B_shift, 
     convA_bias (optional), convB_bias (optional)]
 */
open class ResidualLayer: CompositeLayer {

    static var AConvID = "A"
    static var AConvInstanceNormID = "A-instanceNorm"
    static var BConvID = "B"
    static var BConvInstanceNormID = "B-instanceNorm"
    public var input: NetworkLayer
    public var output: NetworkLayer
    
    public init(device: MTLDevice, convSize: ConvSize, useBias: Bool = false, id: String? = nil) {
        // We need a Dummy here because the Residual starts with a fork. When the network is initialized both of the layers on the fork will be connected to the residual's input. The dummy helps with pointer management.
        let resId = id ?? ""
        let residual = Dummy() // gets removed after graph is complete
                       ->> [Identity(id: resId + "ID"),
                            Convolution(convSize: convSize, neuronType: .none, useBias: useBias, id: resId + ResidualLayer.AConvID)
                                ->> InstanceNorm(id: resId + ResidualLayer.AConvInstanceNormID)
                                ->> Neuron(type: .relu, id: resId + "RELU")
                                ->> Convolution(convSize: convSize, neuronType: .none, useBias: useBias, id: resId + ResidualLayer.BConvID)
                                ->> InstanceNorm(id: resId + ResidualLayer.BConvInstanceNormID)]
                        ->> Add(device: device, id: resId + "ADD")

        self.input = residual.input
        self.output = residual.output
    }
    
}

