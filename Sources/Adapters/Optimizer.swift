//
//  Optimizer.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

protocol Optimizer {

    associatedtype Graph: GraphProtocol
    func optimize(graph: Graph) -> Graph

}

public protocol TFOptimizer {

    func optimize(graph: TensorflowGraph)
    
}

extension TFOptimizer {

    func addNeuronIfThere(node: TensorflowNode) {
        let outgoing = node.outgoingNodes()
        if outgoing.count == 1, let next = (outgoing.first as? TensorflowNode),
            next.nodeDef.op.isTFReLuOp || next.nodeDef.op.isTFTanhOp || next.nodeDef.op.isTFSigmoidOp {
            var neuron = Tensorflow_AttrValue()
            neuron.value = Tensorflow_AttrValue.OneOf_Value.s(next.nodeDef.op.data(using: .utf8)!)
            node.nodeDef.attr[Constants.CustomAttr.neuron] = neuron

            for output in next.outgoingNodes() {
                output.addIncomingEdge(from: node)
            }

            next.strip()
        }
    }

}
