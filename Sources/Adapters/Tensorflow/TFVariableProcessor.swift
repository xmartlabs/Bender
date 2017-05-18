//
//  TFVariableProcessor.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

class TFVariableProcessor: TFOptimizer {

    func optimize(graph: TensorflowGraph) -> TensorflowGraph {
        for node in graph.nodes {
            if node.nodeDef.name.isTFVariableValue {
                print(node.nodeDef.name + " is a variable init value")
                if let assign = node.outgoingNodes().first,
                   let variable = assign.incomingNodes().filter({ ($0 as? TensorflowNode)?.nodeDef.name.isTFVariableV2 ?? false }).first,
                   let read = variable.outgoingNodes().filter({ ($0 as? TensorflowNode)?.nodeDef.name.isTFVariableRead ?? false }).first,
                   let outputNode = read.outgoingNodes().first {
                    assign.strip()
                    variable.strip()
                    read.strip()
                    outputNode.addIncomingEdge(from: node)
                }
            }
        }
        return graph
    }

}

fileprivate extension String {

    var isTFVariableValue: Bool {
        let regex = try! Regex("Variable(_\\d+)?/initial_value")
        return regex.test(self)
    }

    var isTFVariableV2: Bool {
        let regex = try! Regex("Variable(_\\d+)?")
        return regex.test(self)
    }

    var isTFVariableRead: Bool {
        let regex = try! Regex("Variable(_\\d+)?/read")
        return regex.test(self)
    }
    
}
