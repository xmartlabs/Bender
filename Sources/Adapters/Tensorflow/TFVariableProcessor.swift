//
//  TFVariableProcessor.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

/// Processes `Variable` nodes / groups from TensorFlow.
public class TFVariableProcessor: TFOptimizer {

    /*  Takes
                Const --> Assign        Output
                            ^             ^
                        VariableV2  -->  Read

        Returns
                Const  -->  VariableV2  -->  Output
     */
    
    public func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if node.nodeDef.isTFVariableV2Op {
                if let assign = node.outgoingNodes().filter({ ($0 as? TFNode)?.nodeDef.isTFVariableAssignOp ?? false }).first,
                   let read = node.outgoingNodes().filter({ ($0 as? TFNode)?.nodeDef.name.isTFVariableReadName ?? false }).first,
                   let outputNode = read.outgoingNodes().first {
                    read.strip()
                    if let constValue = assign.incomingNodes().filter({ ($0 as? TFNode)?.nodeDef.isTFConstOp ?? false }).first {
                        // Embedded const variables
                        assign.strip()
                        outputNode.addIncomingEdge(from: node)
                        node.addIncomingEdge(from: constValue)
                    } else {
                        // No variables or randomly initialized. You must pass a parameter loader in this case
                        assign.deleteIncomingEdge(node: node)
                        assign.strip(recursive: true)
                        outputNode.addIncomingEdge(from: node)
                    }
                }
            } else if node.nodeDef.name.isTFVariableReadName,
                    let readInput = node.incomingNodes() as? [TFNode], readInput.count == 1,
                    let variable = readInput.first, variable.nodeDef.isTFConstOp, variable.incomingNodes().isEmpty,
                    let outputs = node.outgoingNodes() as? [TFNode], outputs.count == 1 , let output = outputs.first {
                //Freezed graph variables are transformed to Const

                node.strip()
                output.addIncomingEdge(from: variable)
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
        let regex = try! Regex(".*/read$")
        return regex.test(self)
    }
    
}
