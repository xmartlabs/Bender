//
//  Constants.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 3/14/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import CoreVideo
import Metal

struct Constants {

    static let FloatSize = MemoryLayout<Float>.size
    static let HalfSize = MemoryLayout<Float>.size / 2

}

extension Constants {

    struct Ops {

        static let Conv = "Conv2D"
        static let Dense = "Dense"
        static let Const = "Const"
        static let Variable = "VariableV2"
        static let MatMul = "MatMul"
        static let Reshape = "Reshape"
        static let BiasAdd = "BiasAdd"
        static let Relu = "Relu"
        static let Tanh = "Tanh"
        static let Sigmoid = "Sigmoid"
        static let Shape = "Shape"

    }

    struct CustomAttr {

        static let neuron = "neuron"
        
    }

}
