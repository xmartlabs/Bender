//
//  TFVariableProcessor.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

class TFVariableProcessor: TFOptimizer {

    /*  Takes
        initial_value --> Assign        Output
                            ^             ^
                        VariableV2  -->  Read

        Returns
        initial_value  -->  VariableV2  -->  Output
     */
    
    func optimize(graph: TensorflowGraph) {
        for node in graph.nodes {
            if node.nodeDef.op.isTFVariableV2Op {
                if let assign = node.outgoingNodes().filter({ ($0 as? TensorflowNode)?.nodeDef.op.isTFVariableAssignOp ?? false }).first,
                   let constValue = assign.incomingNodes().filter({ ($0 as? TensorflowNode)?.nodeDef.op.isTFConstOp ?? false }).first,
                   let read = node.outgoingNodes().filter({ ($0 as? TensorflowNode)?.nodeDef.name.isTFVariableReadName ?? false }).first,
                   let outputNode = read.outgoingNodes().first {
                    assign.strip()
                    read.strip()
                    outputNode.addIncomingEdge(from: node)
                    node.addIncomingEdge(from: constValue)
                }
            }

            //TODO: handle variables that are not in the graph
        }
    }

}

fileprivate extension String {

    var isTFVariableValue: Bool {
        let regex = try! Regex("Variable(_\\d+)?/initial_value")
        return regex.test(self)
    }

    var isTFVariableV2Name: Bool {
        let regex = try! Regex("Variable(_\\d+)?")
        return regex.test(self)
    }

    var isTFVariableReadName: Bool {
        let regex = try! Regex(".*/read")
        return regex.test(self)
    }

    var isTFVariableAssignOp: Bool {
        return self == "Assign"
    }
    
}
