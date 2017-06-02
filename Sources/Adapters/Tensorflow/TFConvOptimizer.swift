//
//  TFConvOptimizer.swift
//  Bender
//
//  Created by Mathias Claassen on 5/23/17.
//
//

import Foundation

/// Combines Conv2d with BiasAdd. Should be executed after Variable Processor
class TFConvOptimizer: TFOptimizer {

    /*  Takes:
        Conv2D --> BiasAdd (or Add) [--> Neuron]
          ^           ^
        Variable    Variable

     Returns:
        Variable -> Conv2D(+add-ons) <- Variable

     */
    func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if node.nodeDef.isTFConvOp,
                let out = node.outgoingNodes() as? [TFNode], out.count == 1,
                let biasAdd = out.first {
                if biasAdd.nodeDef.isTFBiasAddOp || biasAdd.nodeDef.isTFAddOp,
                    let biasVar = (biasAdd.incomingNodes() as? [TFNode])?.first(where: { $0.nodeDef.isTFVariableOrConstOp }) {
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
