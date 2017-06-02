//
//  TFConverter.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

/// Converts a TFNode to a NetworkLayer of Bender
public typealias TFMapper = (TFNode) -> NetworkLayer

/// Converts a TFGraph to a Bender neural network graph.
open class TFConverter: Converter {

    /// Optimizers are the funtions that preprocess and simplify a TFGraph
    public var optimizers: [TFOptimizer]

    /// Dictionary of TFMappers. Keys are the ops used in the nodes of a TensorFlow model
    public var mappers = [String: TFMapper]()

    /// If the framework should print information while converting a graph set this to true.
    public var verbose = false

    public init(optimizers: [TFOptimizer]) {
        self.optimizers = optimizers
    }

    /// Creates a TFConverter with the default optimizers.
    open static func `default`() -> TFConverter {
        let instance = TFConverter(optimizers: [TFStripTrainingOps(),
                                                TFDeleteDropout(),
                                                TFVariableProcessor(),
                                                TFDenseSubstitution(),
                                                TFReshapeOptimizer(),
                                                TFConvOptimizer()])
        instance.setupMappers()
        return instance
    }

    /// Converts a graph from an URL.
    /// - Parameters:
    ///   - file: The file where the TensorFlow graph is stored
    ///   - type: If the file is in text or binary format
    /// - Returns: The converted network layers
    open func convertGraph(file: URL, type: ProtoFileType) -> [NetworkLayer] {
        let loader = TFGraphLoader()
        var graph = loader.load(file: file, type: type)

        if verbose {
            debugPrint("\n\n\nNodes in my graph (pre optimization):")
            for node in graph.nodes {
                debugPrint("\(node.nodeDef.name)")
            }
        }

        runOptimizers(graph: &graph)
        return translateOperators(graph: graph)
    }

    /// Runs all the optimizers on the graph passed as argument.
    /// All nodes that have neither incoming nor outgoing edges are removed.
    public func runOptimizers(graph: inout TFGraph) {
        for optimizer in optimizers {
            // Run optimization
            optimizer.optimize(graph: graph)

            // Remove nodes that are not connected to any other
            graph.removeLonely()
        }

        if verbose {
            debugPrint("\n\n\nNodes in my graph (after optimization):")
            for node in graph.nodes {
                debugPrint("\(node.nodeDef.name)")
            }
        }
    }

    /// Runs the mappers through all the nodes in the `graph`.
    /// Ops that cannot be mapped are discarded. If these ops are in the main path of the graph then the resulting graph will be disconnected. 
    ///
    /// - Parameter graph: The TFGraph to be mapped
    /// - Returns: An array of mapped oprations as NetworkLayer's
    public func translateOperators(graph: TFGraph) -> [NetworkLayer] {
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
                // We found an unsupported layer. We ignore it but warn.
                if verbose {
                    debugPrint("Bender:: Unsupported layer found: \(node.nodeDef.op)")
                }
            }
        }

        return layers
    }
}
