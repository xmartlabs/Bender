//
//  GraphLoader.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/17/17.
//
//

import SwiftProtobuf

class TensorflowGraphLoader {

    func load(file: URL, type: ProtoFileType) -> TensorflowGraph {
        switch type {
        case .binary:
            let data = try! Data(contentsOf: file)
            let graphDef = try! Tensorflow_GraphDef(serializedData: data)
            return TensorflowGraph(graphDef: graphDef)
        case .text:
            let text = try! String(contentsOf: file, encoding: String.Encoding.utf8)
            let graphDef = try! Tensorflow_GraphDef(textFormatString: text)
            return TensorflowGraph(graphDef: graphDef)
        }
    }

}

public class TensorflowGraph: GraphProtocol {

    public var nodes: [TensorflowNode]
    var graphDef: Tensorflow_GraphDef

    init(graphDef: Tensorflow_GraphDef) {
        self.graphDef = graphDef
        self.nodes = []
        self.initNodes()
    }

    func initNodes() {
        var nodesByName = [String: TensorflowNode]()
        self.nodes = graphDef.node.map {
            let node = TensorflowNode(def: $0)
            nodesByName[$0.name] = node
            return node
        }

        for node in nodes {
            for input in node.nodeDef.input {
                if let inputNode = nodesByName[input] {
                    node.addIncomingEdge(from: inputNode)
                } else {
                    print("Unknown input: \(input)")
                }
            }
        }

        // cannot call mutating func on self?
        var me = self
        me.sortNodes()
    }

}

public class TensorflowNode: Node {

    public var edgeIn: [WeakNodeClosure] = []
    public var edgeOut: [Node] = []
    var nodeDef: Tensorflow_NodeDef

    init(def: Tensorflow_NodeDef) {
        self.nodeDef = def
    }

    public func isEqual(to other: Node) -> Bool {
        return self.nodeDef == (other as? TensorflowNode)?.nodeDef
    }

}

func ==(lhs: Tensorflow_NodeDef, rhs: Tensorflow_NodeDef) -> Bool {
    return lhs.name == rhs.name
}
