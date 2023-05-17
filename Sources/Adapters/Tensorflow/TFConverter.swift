//
//  TFConverter.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

/// Signals if a file is in binary or text format
public enum ProtoFileType {

    case binary
    case text

}

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

    var ignoredOps = ["Const", "Placeholder"]

    /// Proto buffer type
    fileprivate var type: ProtoFileType!

    public init(type: ProtoFileType, optimizers: [TFOptimizer], verbose: Bool = false) {
        self.type = type
        self.optimizers = optimizers
        self.verbose = verbose
        setupMappers()
    }

    /// Creates a TFConverter with the default optimizers.
    public static func `default`(
        type: ProtoFileType = .binary,
        additionalOptimizers: [TFOptimizer] = [],
        verbose: Bool = false) -> TFConverter {

        let instance = TFConverter(
            type: type,
            optimizers: [
                TFStripTrainingOps(),
                TFIgnoredOpsDeleter(),
                TFDeleteDropout(),
                TFVariableProcessor(),
                TFDenseSubstitution(),
                TFReshapeOptimizer(),
                TFConvDilationOptimizer(),
                TFConvOptimizer()
            ] + additionalOptimizers,
            verbose: verbose
        )
        return instance
    }

    /// Converts a graph from an URL.
    /// - Parameters:
    ///   - file: The file where the TensorFlow graph is stored
    ///   - type: If the file is in text or binary format
    /// - Returns: The converted network layers
    open func convertGraph(file: URL, startNodes: [Start]) -> [NetworkLayer] {
        let loader = TFGraphLoader()
        var graph = loader.load(file: file, type: type)

        if verbose {
            debugPrint("\n\n\nNodes in my graph (pre optimization):")
            for node in graph.nodes {
                debugPrint("\(node.nodeDef.name) \(node.nodeDef.op)")
            }
        }

        runOptimizers(graph: &graph)
        return translateOperators(graph: graph, startNodes: startNodes)
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
    public func translateOperators(graph: TFGraph, startNodes: [Start]) -> [NetworkLayer] {
        var layers = [NetworkLayer]()
        var processed = [String: NetworkLayer]() // maps TF node names to layers (to link nodes)
        
        for node in graph.nodes {
            if let mapper = mappers[node.nodeDef.op] {
                let layer = mapper(node)
                layers.append(layer)
                processed[node.nodeDef.name] = layer
                // I get the indexes for all the input nodes and link the corresponding network layers
                // For this to work we must add the Dummies when we cannot translate an `op`
                for input in node.incomingNodes().cleanMap({ $0 as? TFNode }) {
                    // Following unwrap will never fail because all nodes are TFNode
                    // If clause is false only when the input node could not be processed because we do not support it.
                    // This is because graphs are dependency ordered and therefore dependencies will always be processed before.
                    if let inputLayer = processed[input.nodeDef.name] {
                        layer.addIncomingEdge(from: inputLayer)
                    }
                }
            } else if node.nodeDef.op == Constants.Ops.Placeholder {
                let layer = startNodes.count == 1 ? startNodes.first :
                    startNodes.first(where: { $0.inputName == node.nodeDef.name })
                if let layer = layer {
                    layers.insert(layer, at: 0)
                    processed[node.nodeDef.name] = layer
                }
            } else if !(ignoredOps.contains(node.nodeDef.op)) {
                // We found an unsupported layer. We ignore it but warn.
                if verbose {
                    debugPrint("Bender:: Unsupported layer found: \(node.nodeDef.op)")
                }
            }
        }

        return layers
    }
}
