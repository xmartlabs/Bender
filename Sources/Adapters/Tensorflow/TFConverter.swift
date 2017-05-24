//
//  TFConverter.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

public typealias TFMapper = (TFNode) -> NetworkLayer

public class TFConverter: Converter {

    public var optimizers: [TFOptimizer]
    public var mappers = [String: TFMapper]()

    public init(optimizers: [TFOptimizer]) {
        self.optimizers = optimizers
    }

    public static func `default`() -> TFConverter {
        let instance = TFConverter(optimizers: [TFStripTrainingOps(),
                                                TFDeleteDropout(),
                                                TFVariableProcessor(),
                                                TFDenseSubstitution(),
                                                TFConvOptimizer()])
        instance.setupMappers()
        return instance
    }

    public func convertGraph(file: URL, type: ProtoFileType) -> [NetworkLayer] {
        let loader = TFGraphLoader()
        var graph = loader.load(file: file, type: type)

        debugPrint("\n\n\nNodes in my graph (pre optimization):")
        for node in graph.nodes {
            debugPrint("\(node.nodeDef.name)")
        }

        runOptimizers(graph: &graph)
        return translateOperators(graph: graph)
    }

    func runOptimizers(graph: inout TFGraph) {
        for optimizer in optimizers {
            optimizer.optimize(graph: graph)
            graph.removeLonely()
        }

        debugPrint("\n\n\nNodes in my graph (after optimization):")
        for node in graph.nodes {
            debugPrint("\(node.nodeDef.name)")
        }
    }

    func translateOperators(graph: TFGraph) -> [NetworkLayer] {
        var layers = [NetworkLayer]()
        var processed = [String: NetworkLayer]() // maps TF node names to layers (to link nodes)

        for node in graph.nodes {
            if let mapper = mappers[node.nodeDef.op] {
                let layer = mapper(node)
                layers.append(layer)
                processed[node.nodeDef.name] = layer
                // I get the indexes for all the input nodes and link the corresponding network layers
                // For this to work we must add the Dummies when we cannot translate an `op`
                for input in node.incomingNodes() {
                    // Following unwrap will never fail because all nodes are TFNode
                    // If clause is false only when the input node could not be processed because we do not support it.
                    // This is because graphs are dependency ordered and therefore dependencies will always be processed before.
                    if let inputLayer = processed[(input as! TFNode).nodeDef.name] {
                        layer.addIncomingEdge(from: inputLayer)
                    }
                }
            } else {
                // We found an unsupported layer. We ignore it but warn and add a dummy to maintain the indexes of the nodes
                debugPrint("Palladium:: Unsupported layer found: \(node.nodeDef.op)")
            }
        }

        return layers
    }
}
