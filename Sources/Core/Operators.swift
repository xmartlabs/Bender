//
//  Operators.swift
//  Palladium
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

@discardableResult
public func ->> (left: Group, right: Group) -> Group {
    right.input.addIncoming(layer: left.output)
    let group = LayerGroup(input: left.input, output: right.output)
    return group
}

@discardableResult
public func ->> (left: [Group], right: Group) -> Group {
    for layer in left {
        right.input.addIncoming(layer: layer.output)
    }
    return LayerGroup(input: left[0].input, output: right.output)
}

@discardableResult
public func ->> (left: Group, right: [Group]) -> [Group] {
    var output = [Group]()
    for layer in right {
        layer.input.addIncoming(layer: left.output)
        output.append(LayerGroup(input: left.input, output: layer.output))
    }
    return output
}
