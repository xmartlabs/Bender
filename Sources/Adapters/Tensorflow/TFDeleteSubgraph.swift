//
//  TFDeleteDropout.swift
//  Bender
//
//  Created by Mathias Claassen on 5/23/17.
//
//

import Foundation

/// Deletes a specific subgraph from a TFGraph
public protocol TFDeleteSubgraphOptimizer: TFOptimizer {

    /// This Regex tells if a node is in a subgraph to be deleted or not. 
    /// If the node's name has a match for this regex then it will be considered as belonging to a subgraph
    var regex: Regex { get set }

    /// Called for all nodes in the subgraph to be deleted
    /// - Returns: If the node is connected to an input that should be rewired
    func isInputNode(_ node: TFNode) -> Bool

    /// Called for all nodes in the subgraph to be deleted
    /// - Returns: If the node is connected to an output that should be rewired
    func isOutputNode(_ node: TFNode) -> Bool

}

public extension TFDeleteSubgraphOptimizer {

    /// Tells if a node is in the subgraph or not
    func isInSubgraph(_ node: TFNode) -> Bool {
        return regex.test(node.nodeDef.name)
    }

    /// Returns an identifier for a node in this graph
    func id(for node: TFNode) -> String {
        let match = regex.match(node.nodeDef.name)
        return (node.nodeDef.name as NSString).substring(to: match.location + match.length)
    }

    /// Returns if the node has incoming connections to nodes outside of the subgraph
    func isInputNode(_ node: TFNode) -> Bool {
        // If the subgraph should be discarded without rewiring
        return false
    }

    /// Returns if the node has outgoing connections to nodes outside of the subgraph
    func isOutputNode(_ node: TFNode) -> Bool {
        // If the subgraph should be discarded without rewiring
        return false
    }

    func optimize(graph: TFGraph) {
        var mappings = [String: (inputs: [TFNode]?, outputs: [TFNode]?)]()
        for node in graph.nodes where isInSubgraph(node) {
            if isInputNode(node) {
                if let inputs = (node.incomingNodes() as? [TFNode])?.filter({ !isInSubgraph($0) }) {
                    var currentValue = mappings[id(for: node)] ?? (nil, nil)
                    if currentValue.inputs == nil {
                        currentValue.inputs = inputs
                    } else {
                        currentValue.inputs?.append(contentsOf: inputs)
                    }
                    mappings[id(for: node)] = currentValue
                }
            } else if isOutputNode(node) {
                if let outputs = (node.outgoingNodes() as? [TFNode])?.filter({ !isInSubgraph($0) }) {
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
