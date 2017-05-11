//
//  ComputePipelineDef.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/10/17.
//
//

import Foundation

struct ComputePipelineDef: Hashable {

    var hashValue: Int {
        return function.hashValue
    }
    var function: String
    var constants: [FunctionConstantBase]

}

func == (left: ComputePipelineDef, right: ComputePipelineDef) -> Bool {
    return left.function == right.function && left.constants == right.constants
}
