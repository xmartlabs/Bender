//
//  TFDeleteSave.swift
//  Bender
//
//  Created by Mathias Claassen on 5/22/17.
//
//

import Foundation

/// Strips common nodes tht are used in training but not in evaluating/testing
public class TFStripTrainingOps: TFOptimizer {

    public var regexes: [Regex] = [TFDeleteSave().regex, TFDeleteRegularizer().regex, TFDeleteInitializer().regex]

    public func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if let _ = regexes.first(where: { $0.test(node.nodeDef.name) }) {
                node.strip()
            }
        }
    }

}

/// Deletes 'Save' subgraphs
public class TFDeleteSave: TFDeleteSubgraphOptimizer {

    public var regex: Regex = try! Regex("save(_\\d+)?/")
    
}

/// Deletes 'Initializer' subgraphs
public class TFDeleteInitializer: TFDeleteSubgraphOptimizer {

    public var regex: Regex = try! Regex("Initializer(_\\d+)?/")

}

/// Deletes 'Regularizer' subgraphs
public class TFDeleteRegularizer: TFDeleteSubgraphOptimizer {

    public var regex: Regex = try! Regex("Regularizer(_\\d+)?/")

}

/// Deletes 'Dropout' subgraphs
public class TFDeleteDropout: TFDeleteSubgraphOptimizer {

    public var regex: Regex = try! Regex("dropout(_\\d+)?/")

    public func isInputNode(_ node: TFNode) -> Bool {
        return node.nodeDef.isTFShapeOp
    }

    public func isOutputNode(_ node: TFNode) -> Bool {
        return node.nodeDef.name.isTFDropoutMulName
    }

}

fileprivate extension String {

    var isTFDropoutMulName: Bool {
        let regex = try! Regex("dropout(_\\d+)?/mul")
        return regex.test(self)
    }
    
}
