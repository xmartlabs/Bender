//
//  Operators.swift
//  Bender
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import Foundation

precedencegroup AppendPrecedence {
    associativity: left
    higherThan: LogicalConjunctionPrecedence
}

infix operator ->> : AppendPrecedence

/// Combines two groups of nodes into one by linking the last node of the left element to the first of the right
@discardableResult
public func ->> (left: Group, right: Group) -> Group {
    right.input.addIncomingEdge(from: left.output)
    let group = LayerGroup(input: left.input, output: right.output)
    return group
}

/// Combines a list of groups of nodes (left) to a single union group (right). Links the last node of each group in left to the first node of right
@discardableResult
public func ->> (left: [Group], right: Group) -> Group {
    for layer in left {
        right.input.addIncomingEdge(from: layer.output)
    }
    //TODO: handle different inputs with a Dummy node.
    return LayerGroup(input: left[0].input, output: right.output)
}

/// Combines a group of nodes (left) to a list of in parallel executed groups of nodes (right).
/// Links the last node of left to the first node of each group in right
@discardableResult
public func ->> (left: Group, right: [Group]) -> [Group] {
    var output = [Group]()
    for layer in right {
        layer.input.addIncomingEdge(from: left.output)
        output.append(LayerGroup(input: left.input, output: layer.output))
    }
    return output
}
