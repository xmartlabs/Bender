//
//  TFDeleteDropout.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/23/17.
//
//

import Foundation

protocol TFDeleteSubgraphOptimizer: TFOptimizer {

    /// This Regex tells if a node is in a subgraph to be deleted or not. 
    /// If the node's name has a match for this regex then it will be considered as belonging to a subgraph
    var regex: Regex { get set }

    /// Called for all nodes in the subgraph to be deleted
    /// - Returns: If the node is connected to an input that should be rewired
    func isInputNode(_ node: TensorflowNode) -> Bool

    /// Called for all nodes in the subgraph to be deleted
    /// - Returns: If the node is connected to an output that should be rewired
    func isOutputNode(_ node: TensorflowNode) -> Bool

}

extension TFDeleteSubgraphOptimizer {

    func isInSubgraph(_ node: TensorflowNode) -> Bool {
        return regex.test(node.nodeDef.name)
    }

    func id(for node: TensorflowNode) -> String {
        let match = regex.match(node.nodeDef.name)
        return (node.nodeDef.name as NSString).substring(to: match.location + match.length)
    }

    func isInputNode(_ node: TensorflowNode) -> Bool {
        // If the subgraph should be discarded without rewiring
        return false
    }

    func isOutputNode(_ node: TensorflowNode) -> Bool {
        // If the subgraph should be discarded without rewiring
        return false
    }

    func optimize(graph: TensorflowGraph) {
        var mappings = [String: (inputs: [TensorflowNode]?, outputs: [TensorflowNode]?)]()
        for node in graph.nodes {
            if isInSubgraph(node) {
                if isInputNode(node) {
                    if let inputs = (node.incomingNodes() as? [TensorflowNode])?.filter({ !isInSubgraph($0) }) {
                        var currentValue = mappings[id(for: node)] ?? (nil, nil)
                        if currentValue.inputs == nil {
                            currentValue.inputs = inputs
                        } else {
                            currentValue.inputs?.append(contentsOf: inputs)
                        }
                        mappings[id(for: node)] = currentValue
                    }
                } else if isOutputNode(node) {
                    if let outputs = (node.outgoingNodes() as? [TensorflowNode])?.filter({ !isInSubgraph($0) }) {
                        var currentValue = mappings[id(for: node)] ?? (nil, nil)
                        if currentValue.outputs == nil {
                            currentValue.outputs = outputs
                        } else {
                            currentValue.outputs?.append(contentsOf: outputs)
                        }
                        mappings[id(for: node)] = currentValue
                    }
                }
                node.strip()
            }
        }

        // wire together
        for id in mappings.keys {
            if let inputs = mappings[id]?.inputs, let outputs = mappings[id]?.outputs {
                for output in outputs {
                    for input in inputs {
                        output.addIncomingEdge(from: input)
                    }
                }
            }
        }
    }
    
}
