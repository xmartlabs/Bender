//
//  TFConvOptimizer.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/23/17.
//
//

import Foundation

/// Combines Conv2d with BiasAdd. Should be executed after Variable Processor
class TFConvOptimizer: TFOptimizer {

    /*  Takes:
        Conv2D --> BiasAdd [--> Neuron]
          ^           ^
        Variable    Variable

     Returns:
        Variable -> Conv2D(+add-ons) <- Variable

     */
    func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if node.nodeDef.op.isTFConvOp,
                let out = node.outgoingNodes() as? [TFNode], out.count == 1,
                let biasAdd = out.first {
                if biasAdd.nodeDef.op.isTFBiasAddOp,
                    let biasVar = (biasAdd.incomingNodes() as? [TFNode])?.filter({ $0.nodeDef.op.isTFVariableV2Op }).first {
                    node.addIncomingEdge(from: biasVar)
                    for output in biasAdd.outgoingNodes() {
                        output.addIncomingEdge(from: node)
                    }
                    biasAdd.strip()
                }

                // add neuron data
                addNeuronIfThere(node: node)
            }
        }
    }
    
}
