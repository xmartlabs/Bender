//
//  TFNode.swift
//  Bender
//
//  Created by Mathias Claassen on 5/24/17.
//
//

import Foundation

/// A node of a graph imported from TensorFlow
public class TFNode: Node {

    public var edgeIn: [WeakNodeClosure] = []
    public var edgeOut: [Node] = []

    /// TensorFlow node definition
    public var nodeDef: Tensorflow_NodeDef

    public init(def: Tensorflow_NodeDef) {
        self.nodeDef = def
    }

    public func isEqual(to other: Node) -> Bool {
        return self.nodeDef == (other as? TFNode)?.nodeDef
    }

}

public func ==(lhs: Tensorflow_NodeDef, rhs: Tensorflow_NodeDef) -> Bool {
    return lhs.name == rhs.name
}
