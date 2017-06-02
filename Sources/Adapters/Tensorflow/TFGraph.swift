//
//  TFGraph.swift
//  Bender
//
//  Created by Mathias Claassen on 5/24/17.
//
//

import Foundation

/// Represents a graph that is imported from a TensorFlow model
public class TFGraph: GraphProtocol {

    /// Nodes in this graph
    public var nodes: [TFNode]

    /// TensorFlow graph definition
    public var graphDef: Tensorflow_GraphDef

    public init(graphDef: Tensorflow_GraphDef) {
        self.graphDef = graphDef
        self.nodes = []
        self.initNodes()
    }

    public func initNodes() {
        var nodesByName = [String: TFNode]()
        self.nodes = graphDef.node.map {
            let node = TFNode(def: $0)
            nodesByName[$0.name] = node
            return node
        }

        for node in nodes {
            for input in node.nodeDef.input {
                if let inputNode = nodesByName[input] {
                    node.addIncomingEdge(from: inputNode)
                } else {
                    debugPrint("Unknown input: \(input)")
                }
            }
        }

        // cannot call mutating func on self?
        var me = self
        me.sortNodes()
    }
    
}
