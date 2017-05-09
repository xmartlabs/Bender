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

    public var input: NetworkLayer
    public var output: NetworkLayer
    
    public init(device: MTLDevice, convSize: ConvSize, id: String? = nil, weightFiles: String...) {
        assert(weightFiles.count == 6 || weightFiles.count == 8)
        var biasA: String?
        var biasB: String?
        if weightFiles.count == 8 {
            biasA = weightFiles[6]
            biasB = weightFiles[7]
        }

        let residual = Dummy()
                       ->> [Identity(id: "resID"),
                            Convolution(convSize: convSize, neuronType: .relu, weightsFile: weightFiles[0], biasFile: biasA, id: "resConvA")
                            ->> InstanceNorm(scaleFile: weightFiles[1], shiftFile: weightFiles[2], id: "resINormA")
                            ->> Neuron(type: .relu, id: "resRelu")
                            ->> Convolution(convSize: convSize, neuronType: .none, weightsFile: weightFiles[3], biasFile: biasB, id: "resConvB")
                            ->> InstanceNorm(scaleFile: weightFiles[4], shiftFile: weightFiles[5], id: "resINormB")]
                        ->> Add(device: device, id: "resADD")

        self.input = residual.input
        self.output = residual.output
    }
    
}

