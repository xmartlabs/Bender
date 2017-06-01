//
//  Optimizer.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

public protocol TFOptimizer {

    func optimize(graph: TFGraph)
    
}

extension TFOptimizer {

    func addNeuronIfThere(node: TFNode) {
        let outgoing = node.outgoingNodes()
        if outgoing.count == 1, let next = (outgoing.first as? TFNode),
            next.nodeDef.isTFReLuOp || next.nodeDef.isTFTanhOp || next.nodeDef.isTFSigmoidOp {
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
