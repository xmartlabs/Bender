//
//  TFDeleteSave.swift
//  Bender
//
//  Created by Mathias Claassen on 5/22/17.
//
//

/// Strips common nodes that are used in training but not in evaluating/testing

public class TFStripTrainingOps: TFOptimizer {

    public var regexes: [Regex] = [TFDeleteSave().regex, TFDeleteRegularizer().regex, TFDeleteInitializer().regex]

    public func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if regexes.first(where: { $0.test(node.nodeDef.name) }) != nil {
                node.strip()
            }
        }
    }

}

public class TFIgnoredOpsDeleter: TFOptimizer {

    let ops = ["NoOp", "ExpandDims", "Cast", "Squeeze", "StopGradient", "CheckNumerics", "Assert", "Equal", "All",
               "Dequantize", "RequantizationRange", "Requantize", "PlaceholderWithDefault", "Identity"]

    public func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if ops.contains(node.nodeDef.op) {
                node.removeFromGraph()
            }
        }
    }

}

// swiftlint:disable force_try

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

// swiftlint:enable force_try
