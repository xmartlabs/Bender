//
//  Converter.swift
//  Bender
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

/// Responsible for converting a model from another framework to Bender
public protocol Converter {

    /// Converts a model from another framework to Bender
    func convertGraph(file: URL, startNodes: [Start]) -> [NetworkLayer]

}
