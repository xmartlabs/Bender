//
//  Graph.swift
//  Bender
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
        let inputs: [T] = nodes.filter { $0.incomingNodes().count == 0 }
        let sorted = DependencyListBuilder().list(from: inputs)
        assert(nodes.count == sorted.count, "Seems you might have a cyclic dependency in your graph. That is not supported!")
        nodes = sorted
    }

}
