//
//  TFConverter.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

public class TFConverter: Converter {

    public var optimizers: [TFOptimizer]

    public init(optimizers: [TFOptimizer]) {
        self.optimizers = optimizers
    }

    public static func `default`() -> TFConverter {
        return TFConverter(optimizers: [TFVariableProcessor()])
    }

    public func convertGraph(file: URL, type: ProtoFileType) -> [NetworkLayer] {
        let loader = TensorflowGraphLoader()
        var graph = loader.load(file: file, type: type)
        print("\n\n\nNodes in my graph (pre optimization):")
        for node in graph.nodes {
            print("\(node.nodeDef.name)")
        }

        for optimizer in optimizers {
            graph = optimizer.optimize(graph: graph)
            graph.removeLonely()
        }

        print("\n\n\nNodes in my graph (after optimization):")
        for node in graph.nodes {
            print("\(node.nodeDef.name)")
        }
        
        //TODO: translate ops
        
        return []
    }

}
