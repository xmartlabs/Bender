//
//  Optimizer.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

/// Processes a grpah imported from TensorFlow applying some optimizations/simplifications
public protocol TFOptimizer {

    /// Optimize a grsph imported from TensorFlow. Nodes that are to be removed should be left without adjacencies
    func optimize(graph: TFGraph)
    
}

public extension TFOptimizer {

    /// Adds "Neuron" information to a node if the outgoing edge node is of a known neuron type.
    /// This information can later be used by the 'activationNeuron' function
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
