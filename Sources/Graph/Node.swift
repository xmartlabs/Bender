//
//  Node.swift
//  Bender
//
//  Created by Mathias Claassen on 5/17/17.
//
//

// This must be a class because a Weak object requires its value to be a class and Equatable

/// Closures that must return weak references to Node
public typealias WeakNodeClosure = () -> (Node?)

/// Element of a Graph
public protocol Node: class {

    /// Incoming connections / adjacencies
    var edgeIn: [WeakNodeClosure] { get set }

    /// Outgoing connections / adjacencies
    var edgeOut: [Node] { get set }

    /// We cannot add Equatable to Node but we can add this. Must return if both objects are equal
    func isEqual(to: Node) -> Bool

}

public extension Node {

    /// Returns if the node has no connections
    var isLonely: Bool {
        return edgeIn.isEmpty && edgeOut.isEmpty
    }

    /// Creates a closure that returns a weak reference to an object
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
            if !node.edgeOut.contains(where: { $0.isEqual(to: self) } ) {
                node.edgeOut.append(self)
            }
        }
    }

    /// Replaces the `old` incoming node by `new`. Also replaces the outgoing
    /// occurrence of `old` in the `new` node.
    func replace(incomingEdge old: Node, with new: Node) {
        let incoming = incomingNodes()
        if let index = incoming.index(where: { $0.isEqual(to: old) }) {
            edgeIn[index] = captureWeakly(object: new)
            if let outIndex = new.edgeOut.index(where: { $0.isEqual(to: old)}) {
                new.edgeOut[outIndex] = self
            }
        }
    }

    /// All incoming connections
    func incomingNodes() -> [Node] {
        return edgeIn.flatMap { $0() }
    }

    /// Delete an incoming connection
    func deleteIncomingEdge(node: Node) {
        if let index = edgeIn.index(where: { $0()?.isEqual(to: node) ?? false } ) {
            _ = edgeIn.remove(at: index)
        }
    }

    /// All outgoing connections
    func outgoingNodes() -> [Node] {
        return edgeOut
    }

    /// Delete an outgoing connection
    func deleteOutgoingEdge(node: Node) {
        if let index = edgeOut.index(where: { $0.isEqual(to: node) } ) {
            edgeOut.remove(at: index)
        }
    }

    /// Removes all edges from this node and rewires the inputs to the outputs
    func removeFromGraph() {
        //TODO: we could combine this function with `strip`
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

    /// Removes all edges of this node. If recursive then it will call strip recursively on all of its neighbors 
    /// (so the calling node should remove himself before calling this recursively)
    func strip(recursive: Bool = false) {
        let outgoing = outgoingNodes()
        let incoming = incomingNodes()
        for out in outgoing {
            out.deleteIncomingEdge(node: self)
        }

        for inc in incoming {
            inc.deleteOutgoingEdge(node: self)
        }

        if recursive {
            for node in incoming {
                node.strip(recursive: true)
            }
            for node in outgoing {
                node.strip(recursive: true)
            }
        }

        edgeOut = []
        edgeIn = []
    }

    /// Will insert all the nodes after this node updating the corresponding links
    /// Precondition: The nodes are already linked among them and there is just one beginning and one end in their graph
    func insert(outgoing nodes: [Node]) {
        guard nodes.count > 0 else {
            return
        }
        for outgoing in outgoingNodes() {
            outgoing.replace(incomingEdge: self, with: nodes.last!)
            self.deleteOutgoingEdge(node: outgoing)
        }
        nodes.first!.addIncomingEdge(from: self)
    }

}
