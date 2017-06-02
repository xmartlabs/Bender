//
//  TFWeightData.swift
//  Bender
//
//  Created by Mathias Claassen on 5/19/17.
//
//

import Foundation

/// Internal struct used to pass weights and bias data around
struct TFWeightData {

    let weights: Data?
    let bias: Data?
    let weightShape: Tensorflow_TensorShapeProto
    let useBias: Bool
    
    static func getWeightData(node: TFNode) -> TFWeightData? {
        let varInputs = node.incomingNodes().filter { ($0 as! TFNode).nodeDef.isTFVariableOrConstOp } as! [TFNode]
        var weightsVar: TFNode
        var biasVar: TFNode?

        if varInputs.count == 1 {
            weightsVar = varInputs[0]
        } else if varInputs.count == 2 {
            if varInputs[0].nodeDef.shape?.isBias == true {
                biasVar = varInputs[0]
                weightsVar = varInputs[1]
            } else if varInputs[1].nodeDef.shape?.isBias == true {
                biasVar = varInputs[1]
                weightsVar = varInputs[0]
            } else {
                fatalError("Conv2D(Transpose) must have 1 or 2 Variable input and one of them must be a bias")
            }
        } else {
            fatalError("Conv2D(Transpose) must have 1 or 2 Variable input")
        }

        guard let shape = weightsVar.nodeDef.shape else {
            fatalError("Conv2D(Transpose) has no shape information")
        }

        var weights: Data?
        var bias: Data?

        if weightsVar.nodeDef.isTFConstOp {
            weights = weightsVar.nodeDef.valueData()
        } else if let weightsNode = weightsVar.incomingNodes().first as? TFNode, weightsNode.nodeDef.isTFConstOp {
            weights = weightsNode.nodeDef.valueData()
        }

        if biasVar?.nodeDef.isTFConstOp == true {
            bias = biasVar?.nodeDef.valueData()
        } else if let biasNode = biasVar?.incomingNodes().first as? TFNode {
            bias = biasNode.nodeDef.valueData()
        }

        return TFWeightData(weights: weights, bias: bias, weightShape: shape, useBias: biasVar != nil)
    }

}
