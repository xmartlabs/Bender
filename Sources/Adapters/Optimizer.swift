//
//  Optimizer.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

protocol Optimizer {

    associatedtype Graph: GraphProtocol
    func optimize(graph: Graph) -> Graph

}

public protocol TFOptimizer {

    func optimize(graph: TensorflowGraph)
    
}
