//
//  TFDenseSubstitution.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/22/17.
//
//

import Foundation

/// Should be executed after Variable Processor
class TFDenseSubstitution: TFOptimizer {

    /*  Takes

     */

    func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if node.nodeDef.op.isTFBiasAddOp,
                let matmul = node.incomingNodes().filter({ ($0 as? TFNode)?.nodeDef.op.isTFMatMulOp ?? false }).first,
                let matmulInputs = matmul.incomingNodes() as? [TFNode],
                let weightVar = matmulInputs.filter({ $0.nodeDef.op.isTFVariableV2Op }).first,
                let input = matmulInputs.filter({ !$0.nodeDef.op.isTFVariableV2Op }).first {

                node.nodeDef.op = Constants.Ops.Dense
                node.addIncomingEdge(from: weightVar)
                if input.nodeDef.op.isTFReshapeOp,
                    let preReshape = (input.incomingNodes() as? [TFNode])?.filter({ !$0.nodeDef.op.isTFConstOp }).first {
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
