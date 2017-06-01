//
//  Converter.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

/// Signals if a file is in binary or text format
public enum ProtoFileType {

    case binary
    case text

}

/// Responsible for converting a model from another framework to Palladium
public protocol Converter {

    /// Converts a model from another framework to Palladium
    func convertGraph(file: URL, type: ProtoFileType) -> [NetworkLayer]

}
