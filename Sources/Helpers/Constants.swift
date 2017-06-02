//
//  Constants.swift
//  Bender
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

    /// TensorFlow ops
    struct Ops {

        static let Add = "Add"
        static let Assign = "Assign"
        static let BiasAdd = "BiasAdd"
        static let Conv = "Conv2D"
        static let Const = "Const"
        static let Dense = "Dense"
        static let InstanceNormAdd = "InstanceNormAdd"
        static let InstanceNormMul = "InstanceNormMul"
        static let MatMul = "MatMul"
        static let Mul = "Mul"
        static let Mean = "Mean"
        static let Pow = "Pow"
        static let RealDiv = "RealDiv"
        static let Relu = "Relu"
        static let Reshape = "Reshape"
        static let Shape = "Shape"
        static let Sigmoid = "Sigmoid"
        static let Sub = "Sub"
        static let Tanh = "Tanh"
        static let Variable = "VariableV2"

    }

    /// Custom attributes added to TensorFlow nodes during graph conversion
    struct CustomAttr {

        static let neuron = "neuron"
        
    }

}
