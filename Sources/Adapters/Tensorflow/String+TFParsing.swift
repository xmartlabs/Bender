//
//  String+TFParsing.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

extension String {

    var isTFAddOp: Bool { return self == Constants.Ops.Add }
    var isTFBiasAddOp: Bool { return self == Constants.Ops.BiasAdd }
    var isTFConvOp: Bool { return self == Constants.Ops.Conv }
    var isTFConstOp: Bool { return self == Constants.Ops.Const }
    var isTFInstanceNormMulOp: Bool { return self == Constants.Ops.InstanceNormMul }
    var isTFMatMulOp: Bool { return self == Constants.Ops.MatMul }
    var isTFMeanOp: Bool { return self == Constants.Ops.Mean }
    var isTFMulOp: Bool { return self == Constants.Ops.Mul }
    var isTFPowOp: Bool { return self == Constants.Ops.Pow }
    var isTFRealDivOp: Bool { return self == Constants.Ops.RealDiv }
    var isTFReshapeOp: Bool { return self == Constants.Ops.Reshape }
    var isTFReLuOp: Bool { return self == Constants.Ops.Relu }
    var isTFShapeOp: Bool { return self == Constants.Ops.Shape }
    var isTFSigmoidOp: Bool { return self == Constants.Ops.Sigmoid }
    var isTFSubOp: Bool { return self == Constants.Ops.Sub }
    var isTFTanhOp: Bool { return self == Constants.Ops.Tanh }
    var isTFVariableV2Op: Bool { return self == Constants.Ops.Variable}

}
