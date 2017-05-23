//
//  TFDeleteOpOptimizer.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

class TFDeleteOpOptimizer: TFOptimizer {

    var prefixes = ["Dropout", "Regularizer", "random_uniform", "truncated_normal", "Save", "Restore"]

    func optimize(graph: TensorflowGraph) {

    }

    private func runLoop(graph: TensorflowGraph, from: Int) {
        for index in from..<graph.nodes.count {
            let node = graph.nodes[index]
            if node.nodeDef.name.components(separatedBy: "/").contains(where: { prefixes.contains($0) }) {

            }

        }
    }


}
