//
//  TFGraph.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/24/17.
//
//

import Foundation


public class TFGraph: GraphProtocol {

    public var nodes: [TFNode]
    var graphDef: Tensorflow_GraphDef

    init(graphDef: Tensorflow_GraphDef) {
        self.graphDef = graphDef
        self.nodes = []
        self.initNodes()
    }

    func initNodes() {
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
