//
//  TFDenseSubstitution.swift
//  Palladium
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
        for node in graph.nodes {
            if node.nodeDef.isTFBiasAddOp,
                let matmul = node.incomingNodes().first(where: { ($0 as? TFNode)?.nodeDef.isTFMatMulOp ?? false }),
                let matmulInputs = matmul.incomingNodes() as? [TFNode],
                let weightVar = matmulInputs.first(where: { $0.nodeDef.isTFVariableOrConstOp }),
                let input = matmulInputs.first(where: { !$0.nodeDef.isTFVariableOrConstOp }) {

                node.nodeDef.op = Constants.Ops.Dense
                node.addIncomingEdge(from: weightVar)
                if input.nodeDef.isTFReshapeOp,
                    let preReshape = (input.incomingNodes() as? [TFNode])?.first(where: { !$0.nodeDef.isTFConstOp }) {
                    node.addIncomingEdge(from: preReshape)
                    input.strip()
                } else {
                    node.addIncomingEdge(from: input)
                }
                matmul.strip()

                // add neuron data
                addNeuronIfThere(node: node)
            }
        }
    }

}
