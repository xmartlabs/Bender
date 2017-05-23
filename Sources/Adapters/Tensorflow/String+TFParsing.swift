//
//  String+TFParsing.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

extension String {

    var isTFConvOp: Bool {
        return self == Constants.Ops.Conv
    }

    var isTFConstOp: Bool {
        return self == Constants.Ops.Const
    }

    var isTFVariableV2Op: Bool {
        return self == Constants.Ops.Variable
    }

    var isTFBiasAddOp: Bool {
        return self == Constants.Ops.BiasAdd
    }

    var isTFMatMulOp: Bool {
        return self == Constants.Ops.MatMul
    }

    var isTFReshapeOp: Bool {
        return self == Constants.Ops.Reshape
    }

    var isTFReLuOp: Bool {
        return self == Constants.Ops.Relu
    }

    var isTFTanhOp: Bool {
        return self == Constants.Ops.Tanh
    }

    var isTFSigmoidOp: Bool {
        return self == Constants.Ops.Sigmoid
    }

    var isTFShapeOp: Bool {
        return self == Constants.Ops.Shape
    }

}
