//
//  ComputePipelineDef.swift
//  Bender
//
//  Created by Mathias Claassen on 5/10/17.
//
//

/// Hash key for MTLComputePipelineState
struct ComputePipelineDef: Hashable {

    func hash(into hasher: inout Hasher) {
        hasher.combine(function)
    }

    /// Name of compute kernel function
    var function: String

    /// Function constants for the compute kernel function
    var constants: [FunctionConstantBase]

    static func == (left: ComputePipelineDef, right: ComputePipelineDef) -> Bool {
        return left.function == right.function && left.constants == right.constants
    }
}
