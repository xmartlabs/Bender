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
        static let AddV2 = "AddV2"
        static let Assign = "Assign"
        static let AvgPool = "AvgPool"
        static let BatchToSpace = "BatchToSpaceND"
        static let BatchNormGlobal = "BatchNormWithGlobalNormalization"
        static let FusedBatchNorm = "FusedBatchNorm"
        static let FusedBatchNormV3 = "FusedBatchNormV3"
        static let BiasAdd = "BiasAdd"
        static let Concat = "ConcatV2"
        static let ConcatV1 = "Concat"
        static let Conv = "Conv2D"
        static let Const = "Const"
        static let Dense = "Dense"
        static let DepthwiseConv = "DepthwiseConv2dNative"
        static let InstanceNormAdd = "InstanceNormAdd"
        static let InstanceNormMul = "InstanceNormMul"
        static let MatMul = "MatMul"
        static let MaxPool = "MaxPool"
        static let Mul = "Mul"
        static let Mean = "Mean"
        static let Placeholder = "Placeholder"
        static let Pow = "Pow"
        static let RealDiv = "RealDiv"
        static let QuantizedConv2D = "QuantizedConv2D"
        static let QuantizedRelu = "QuantizedRelu"
        static let QuantizedReshape = "QuantizedReshape"
        static let QuantizeV2 = "QuantizeV2"
        static let Relu = "Relu"
        static let Relu6 = "Relu6"
        static let Reshape = "Reshape"
        static let Rsqrt = "Rsqrt"
        static let Shape = "Shape"
        static let Sigmoid = "Sigmoid"
        static let Softmax = "Softmax"
        static let SpaceToBatch = "SpaceToBatchND"
        static let Sub = "Sub"
        static let Switch = "Switch"
        static let Tanh = "Tanh"
        static let Variable = "VariableV2"

    }

    /// Custom attributes added to TensorFlow nodes during graph conversion
    struct CustomAttr {

        static let neuron = "neuron"

    }

}
