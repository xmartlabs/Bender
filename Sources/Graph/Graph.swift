//
//  Graph.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/17/17.
//
//

import Foundation

/// Protocol implemented by any Graph
public protocol GraphProtocol {

    associatedtype T: Node
    var nodes: [T] { get set }

}

public extension GraphProtocol {

    /// Removes nodes that have no adjacencies
    mutating func removeLonely() {
        nodes = nodes.filter { !$0.isLonely }
    }

    /// Sort nodes by dependencies
    mutating func sortNodes() {
        var sorted = [T]()
        let inputs: [T] = nodes.filter { $0.incomingNodes().count == 0 }
        for input in inputs {
            buildList(node: input, sorted: &sorted)
        }
        assert(nodes.count == sorted.count, "Seems you might have a cyclic dependency in your graph. That is not supported!")
        nodes = sorted
    }

    /// Builds the dependency list for this graph
    private func buildList(node: T, sorted: inout [T]) {
        // check that all the dependencies have been added
        guard !node.incomingNodes().contains (where: { incoming in
            return !sorted.contains(where: { $0.isEqual(to: incoming) } )
        }) else { return }
        sorted.append(node)
        for node in node.outgoingNodes() {
            buildList(node: node as! T, sorted: &sorted)
        }
    }

}
