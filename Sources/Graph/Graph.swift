//
//  Graph.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/17/17.
//
//

import Foundation

public protocol GraphProtocol {

    associatedtype T: Node
    var nodes: [T] { get set }
//    var first: Node<T> { get }
//    var last: Node<T> { get }

//    func remove(node: T)
//    func sortNodes()

}

public extension GraphProtocol {

    mutating func removeLonely() {
        nodes = nodes.filter { !$0.isLonely }
    }

    mutating func sortNodes() {
        var sorted = [T]()
        let inputs: [T] = nodes.filter { $0.incomingNodes().count == 0 }
        for input in inputs {
            buildGraph(node: input, sorted: &sorted)
        }
        assert(nodes.count == sorted.count, "Seems you might have a cyclic dependency in your graph. That is not supported!")
        nodes = sorted
    }

    private func buildGraph(node: T, sorted: inout [T]) {
        // check that all the dependencies have been added
        guard !node.incomingNodes().contains (where: { incoming in
            return !sorted.contains(where: { $0.isEqual(to: incoming) } )
        }) else { return }
        sorted.append(node)
        for node in node.outgoingNodes() {
            buildGraph(node: node as! T, sorted: &sorted)
        }
    }

}
