//
//  Node.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/17/17.
//
//

import Foundation

// This must be a class because a Weak object requires its value to be a class and Equatable

/// Closures that must return weak references to Node
public typealias WeakNodeClosure = () -> (Node?)

public protocol Node: class {

    var edgeIn: [WeakNodeClosure] { get set }
    var edgeOut: [Node] { get set }

    /// We cannot add Equatable to Node but we can add this. Must return if both objects are equal
    func isEqual(to: Node) -> Bool

}

extension Node {

    var isLonely: Bool {
        return edgeIn.count == 0 && edgeOut.count == 0
    }

    func captureWeakly(object: Node) -> (WeakNodeClosure) {
        return { [weak object] in
            return object
        }
    }
    /// Should be called to add an edge. Adds both directions of the edge
    func addIncomingEdge(from node: Node) {
        let incoming = incomingNodes()
        if !incoming.contains(where: { $0.isEqual(to: node) } ) {
            edgeIn.append(captureWeakly(object: node))
            if !node.edgeOut.contains(where: { $0 === self } ) {
                node.edgeOut.append(self)
            }
        }
    }

    func incomingNodes() -> [Node] {
        return edgeIn.flatMap { $0() }
    }

    func deleteIncomingEdge(node: Node) {
        if let index = edgeIn.index(where: { $0()?.isEqual(to: node) ?? false } ) {
            _ = edgeIn.remove(at: index)
        }
    }

    func outgoingNodes() -> [Node] {
        return edgeOut
    }

    func deleteOutgoingEdge(node: Node) {
        if let index = edgeOut.index(where: { $0.isEqual(to: node) } ) {
            edgeOut.remove(at: index)
        }
    }

    func removeFromGraph() {
        //TODO: could we combine this funtion with `strip`
        let outgoing = outgoingNodes()
        let incoming = incomingNodes()
        for out in outgoing {
            out.deleteIncomingEdge(node: self)
        }

        for inc in incoming {
            inc.deleteOutgoingEdge(node: self)
            for out in outgoing {
                out.addIncomingEdge(from: inc)
            }
        }

        edgeOut = []
        edgeIn = []
    }

    func strip() {
        for out in outgoingNodes() {
            out.deleteIncomingEdge(node: self)
        }

        for inc in incomingNodes() {
            inc.deleteOutgoingEdge(node: self)
        }

        edgeOut = []
        edgeIn = []
    }

}

//func ==<T: Equatable> (left: Node<T>, right: Node<T>) -> Bool {
//    return left.value == right.value
//}
