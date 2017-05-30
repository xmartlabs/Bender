//
//  TFDeleteSave.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/22/17.
//
//

import Foundation

class TFStripTrainingOps: TFOptimizer {

    var regexes: [Regex] = [TFDeleteSave().regex, TFDeleteRegularizer().regex, TFDeleteInitializer().regex]

    func optimize(graph: TFGraph) {
        for node in graph.nodes {
            if let _ = regexes.first(where: { $0.test(node.nodeDef.name) }) {
                node.strip()
            }
        }
    }

}

class TFDeleteSave: TFDeleteSubgraphOptimizer {

    var regex: Regex = try! Regex("save(_\\d+)?/")
    
}

class TFDeleteInitializer: TFDeleteSubgraphOptimizer {

    var regex: Regex = try! Regex("Initializer(_\\d+)?/")

}

class TFDeleteRegularizer: TFDeleteSubgraphOptimizer {

    var regex: Regex = try! Regex("Regularizer(_\\d+)?/")

}

class TFDeleteDropout: TFDeleteSubgraphOptimizer {

    var regex: Regex = try! Regex("dropout(_\\d+)?/")

    func isInputNode(_ node: TFNode) -> Bool {
        return node.nodeDef.isTFShapeOp
    }

    func isOutputNode(_ node: TFNode) -> Bool {
        return node.nodeDef.name.isTFDropoutMulName
    }

}

fileprivate extension String {

    var isTFDropoutMulName: Bool {
        let regex = try! Regex("dropout(_\\d+)?/mul")
        return regex.test(self)
    }
    
}
