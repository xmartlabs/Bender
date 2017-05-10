//
//  PALFunctionConstant.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/10/17.
//
//

import Metal

public class FunctionConstantBase: Equatable {
    public var index: Int
    public var type: MTLDataType

    public init(index: Int, type: MTLDataType) {
        self.index = index
        self.type = type
    }

    public func getValue() -> Any {
        fatalError("Not implemented")
    }

    public func isEqual(to other: FunctionConstantBase) -> Bool {
        return type == other.type && index == other.index
    }

}

public func == (left: FunctionConstantBase, right: FunctionConstantBase) -> Bool {
    return left.isEqual(to: right)
}

public class PALFunctionConstant<T: Any>: FunctionConstantBase where T: Equatable {

    public var value: T

    public init(index: Int, type: MTLDataType, value: T) {
        self.value = value
        super.init(index: index, type: type)
    }

    public override func getValue() -> Any {
        return value
    }

    public override func isEqual(to other: FunctionConstantBase) -> Bool {
        guard let otherValue = other.getValue() as? T else { return false }
        return type == other.type && index == other.index && value == otherValue
    }
    
}
