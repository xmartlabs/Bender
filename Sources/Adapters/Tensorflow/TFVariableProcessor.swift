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
                Const --> Assign        Output
                            ^             ^
                        VariableV2  -->  Read

        Returns
                Const  -->  VariableV2  -->  Output
     */
    
    func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if node.nodeDef.op.isTFVariableV2Op {
                if let assign = node.outgoingNodes().filter({ ($0 as? TFNode)?.nodeDef.op.isTFVariableAssignOp ?? false }).first,
                   let read = node.outgoingNodes().filter({ ($0 as? TFNode)?.nodeDef.name.isTFVariableReadName ?? false }).first,
                   let outputNode = read.outgoingNodes().first {
                    read.strip()
                    if let constValue = assign.incomingNodes().filter({ ($0 as? TFNode)?.nodeDef.op.isTFConstOp ?? false }).first {
                        // Embedded const variables
                        assign.strip()
                        outputNode.addIncomingEdge(from: node)
                        node.addIncomingEdge(from: constValue)
                    } else {
                        assign.deleteIncomingEdge(node: node)
                        assign.strip(recursive: true)
                        outputNode.addIncomingEdge(from: node)
                    }
                }
            }
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
