//
//  String+TFParsing.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

extension Tensorflow_NodeDef {

    var isTFAddOp: Bool { return op == Constants.Ops.Add }
    var isTFVariableAssignOp: Bool { return op == Constants.Ops.Assign }
    var isTFBiasAddOp: Bool { return op == Constants.Ops.BiasAdd }
    var isTFConvOp: Bool { return op == Constants.Ops.Conv }
    var isTFConstOp: Bool { return op == Constants.Ops.Const }
    var isTFInstanceNormMulOp: Bool { return op == Constants.Ops.InstanceNormMul }
    var isTFMatMulOp: Bool { return op == Constants.Ops.MatMul }
    var isTFMeanOp: Bool { return op == Constants.Ops.Mean }
    var isTFMulOp: Bool { return op == Constants.Ops.Mul }
    var isTFPowOp: Bool { return op == Constants.Ops.Pow }
    var isTFRealDivOp: Bool { return op == Constants.Ops.RealDiv }
    var isTFReshapeOp: Bool { return op == Constants.Ops.Reshape }
    var isTFReLuOp: Bool { return op == Constants.Ops.Relu }
    var isTFShapeOp: Bool { return op == Constants.Ops.Shape }
    var isTFSigmoidOp: Bool { return op == Constants.Ops.Sigmoid }
    var isTFSubOp: Bool { return op == Constants.Ops.Sub }
    var isTFTanhOp: Bool { return op == Constants.Ops.Tanh }
    var isTFVariableV2Op: Bool { return op == Constants.Ops.Variable }
    var isTFVariableOrConstOp: Bool { return isTFVariableV2Op || isTFConstOp }

}
