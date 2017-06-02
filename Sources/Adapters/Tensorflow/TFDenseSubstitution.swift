//
//  TFDenseSubstitution.swift
//  Bender
//
//  Created by Mathias Claassen on 5/22/17.
//
//

import Foundation

/// Transforms a MatMul and a BiasAdd into a FullyConnected. Should be executed after Variable Processor
/// Does not work with embedded weights. Transposing of weights must be done previously on Python side.
public class TFDenseSubstitution: TFOptimizer {

    /*  Takes:
          MatMul --> BiasAdd [--> Neuron]
            ^           ^
        Variable    Variable
     
     Returns:
        Variable -> BiasAdd(+add-ons) <- Variable

     */

    public func optimize(graph: TFGraph) {
        for node in graph.nodes where node.nodeDef.isTFMatMulOp {
            if let add = node.outgoingNodes().flatMap({ $0 as? TFNode }).first(where: { $0.nodeDef.isTFBiasAddOp ||
                                                                                        $0.nodeDef.isTFAddOp }),
                let matmulInputs = node.incomingNodes() as? [TFNode],
                let weightVar = matmulInputs.first(where: { $0.nodeDef.isTFVariableOrConstOp }),
                let input = matmulInputs.first(where: { !$0.nodeDef.isTFVariableOrConstOp }) {

                add.nodeDef.op = Constants.Ops.Dense
                add.addIncomingEdge(from: weightVar)
                if input.nodeDef.isTFReshapeOp,
                    let preReshape = (input.incomingNodes() as? [TFNode])?.first(where: { !$0.nodeDef.isTFConstOp }) {
                    add.addIncomingEdge(from: preReshape)
                    input.strip()
                } else {
                    add.addIncomingEdge(from: input)
                }
                node.strip()

                // add neuron data
                addNeuronIfThere(node: add)
            }
        }
    }

}
