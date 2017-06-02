//
//  TFReshapeOptimizer.swift
//  Bender
//
//  Created by Mathias Claassen on 6/1/17.
//
//

import Foundation

/// Removes Reshape nodes
public class TFReshapeOptimizer: TFOptimizer {

    public func optimize(graph: TFGraph) {
        for node in graph.nodes where node.nodeDef.isTFReshapeOp {
            if let shape = node.incomingNodes().filter({ ($0 as? TFNode)?.nodeDef.isTFConstOp ?? false }).first {
                shape.strip()
                node.removeFromGraph()
            }
        }
    }

}
