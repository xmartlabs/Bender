//
//  String+TFParsing.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

extension String {

    var isTFConstOp: Bool {
        return self == "Const"
    }

    var isTFVariableV2Op: Bool {
        return self == "VariableV2"
    }

}
