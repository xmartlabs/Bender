//
//  FunctionConstant.swift
//  Bender
//
//  Created by Mathias Claassen on 5/10/17.
//
//

import Metal

/// Base class for FunctionConstant. Should never be instantiated. Just a helper for us to store arrays of FunctionConstants of different types
public class FunctionConstantBase: Equatable {

    /// Index of the function constant in the Metal shaders
    public var index: Int

    /// Type of the function constant in the Metal shaders
    public var type: MTLDataType

    public init(index: Int, type: MTLDataType) {
        self.index = index
        self.type = type
    }

    /// Returns the value of the FunctionConstant
    public func getValue() -> Any {
        fatalError("Not implemented")
    }

    /// Returns if two FunctionConstant's are equal. Helper to allow generic inheritance in Equatable.
    public func isEqual(to other: FunctionConstantBase) -> Bool {
        return type == other.type && index == other.index
    }

}

public func == (left: FunctionConstantBase, right: FunctionConstantBase) -> Bool {
    return left.isEqual(to: right)
}

/// Generic class that holds information for Metals function constants.
public class FunctionConstant<T: Any>: FunctionConstantBase where T: Equatable {

    /// Value to be passed to the compute kernels function constant
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
