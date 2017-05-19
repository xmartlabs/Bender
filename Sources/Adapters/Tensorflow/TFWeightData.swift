//
//  TFWeightData.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/19/17.
//
//

import Foundation

struct TFWeightData {

    let weights: UnsafePointer<Float>?
    let bias: UnsafePointer<Float>?
    let weightShape: Tensorflow_TensorShapeProto
    
    static func getWeightData(node: TensorflowNode) -> TFWeightData? {
        let varInputs = node.incomingNodes().filter { ($0 as! TensorflowNode).nodeDef.op.isTFVariableV2Op } as! [TensorflowNode]
        var weightsVar: TensorflowNode
        var biasVar: TensorflowNode?

        if varInputs.count == 1 {
            weightsVar = varInputs[0]
        } else if varInputs.count == 2 {
            if varInputs[0].nodeDef.attr["shape"]?.shape.isBias == true {
                biasVar = varInputs[0]
                weightsVar = varInputs[1]
            } else if varInputs[1].nodeDef.attr["shape"]?.shape.isBias == true {
                biasVar = varInputs[1]
                weightsVar = varInputs[0]
            } else {
                fatalError("Conv2DTranspose must have 1 or 2 Variable input and one of them must be a bias")
            }
        } else {
            fatalError("Conv2DTranspose must have 1 or 2 Variable input")
        }

        guard let shape = weightsVar.nodeDef.shape else {
            fatalError("Conv2DTranspose has no shape information")
        }

        var weights: UnsafePointer<Float>?
        var bias: UnsafePointer<Float>?

        if let weightsNode = weightsVar.incomingNodes().first as? TensorflowNode,
            let data = weightsNode.nodeDef.attr["value"]?.tensor.tensorContent, weightsNode.nodeDef.op.isTFConstOp {
            weights = (data as NSData).bytes.assumingMemoryBound(to: Float.self)
        }

        if let biasNode = biasVar?.incomingNodes().first as? TensorflowNode,
            let data = biasNode.nodeDef.attr["value"]?.tensor.tensorContent, biasNode.nodeDef.op.isTFConstOp {
            debugPrint("FOUND SOME BIAS: being ignored for now")
            bias = (data as NSData).bytes.assumingMemoryBound(to: Float.self)
        }

        return TFWeightData(weights: weights, bias: bias, weightShape: shape)
    }

}
