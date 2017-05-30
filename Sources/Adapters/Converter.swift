//
//  Converter.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

public enum ProtoFileType {

    case binary
    case text

}

public protocol Converter {
    
    func convertGraph(file: URL, type: ProtoFileType) -> [NetworkLayer]

}
