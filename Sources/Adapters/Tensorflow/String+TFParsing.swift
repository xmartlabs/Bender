//
//  String+TFParsing.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

extension Tensorflow_NodeDef {

    var isTFAddOp: Bool { return op == Constants.Ops.Add }
    var isTFVariableAssignOp: Bool { return op == Constants.Ops.Assign }
    var isTFFusedBatchNorm: Bool { return op == Constants.Ops.FusedBatchNorm }
    var isTFBatchNormGlobal: Bool { return op == Constants.Ops.BatchNormGlobal }
    var isTFBatchToSpace: Bool { return op == Constants.Ops.BatchToSpace }
    var isTFBiasAddOp: Bool { return op == Constants.Ops.BiasAdd }
    var isTFConvOp: Bool { return op == Constants.Ops.Conv }
    var isTFConstOp: Bool { return op == Constants.Ops.Const }
    var isTFDepthwiseConvOp: Bool { return op == Constants.Ops.DepthwiseConv }
    var isTFInstanceNormMulOp: Bool { return op == Constants.Ops.InstanceNormMul }
    var isTFMatMulOp: Bool { return op == Constants.Ops.MatMul }
    var isTFMeanOp: Bool { return op == Constants.Ops.Mean }
    var isTFMulOp: Bool { return op == Constants.Ops.Mul }
    var isTFPowOp: Bool { return op == Constants.Ops.Pow }
    var isTFQReshapeOp: Bool { return op == Constants.Ops.QuantizedReshape }
    var isTFRsqrtOp: Bool { return op == Constants.Ops.Rsqrt }
    var isTFRealDivOp: Bool { return op == Constants.Ops.RealDiv }
    var isTFReshapeOp: Bool { return op == Constants.Ops.Reshape }
    var isTFReLuOp: Bool { return op == Constants.Ops.Relu }
    var isTFShapeOp: Bool { return op == Constants.Ops.Shape }
    var isTFSigmoidOp: Bool { return op == Constants.Ops.Sigmoid }
    var isTFSpaceToBatch: Bool { return op == Constants.Ops.SpaceToBatch }
    var isTFSubOp: Bool { return op == Constants.Ops.Sub }
    var isTFSwitchOp: Bool { return op == Constants.Ops.Switch }
    var isTFTanhOp: Bool { return op == Constants.Ops.Tanh }
    var isTFVariableV2Op: Bool { return op == Constants.Ops.Variable }
    var isTFVariableOrConstOp: Bool { return isTFVariableV2Op || isTFConstOp }

}

// Node Names
extension Tensorflow_NodeDef {

    var isTFMovMean: Bool { return name.hasSuffix("/moving_mean") }
    var isTFMovVariance: Bool { return name.hasSuffix("/moving_variance") }
    var isTFGamma: Bool { return name.hasSuffix("/gamma") }
    var isTFBeta: Bool { return name.hasSuffix("/beta") }

}
