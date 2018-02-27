//
//  GraphLoader.swift
//  Bender
//
//  Created by Mathias Claassen on 5/17/17.
//
//

import Foundation
import SwiftProtobuf

/// Imports a graph from TensorFlow
class TFGraphLoader {

    func load(file: URL, type: ProtoFileType) -> TFGraph {
        do {
            switch type {
            case .binary:
                let data = try Data(contentsOf: file)
                let graphDef = try Tensorflow_GraphDef(serializedData: data)
                return TFGraph(graphDef: graphDef)
            case .text:
                let text = try String(contentsOf: file, encoding: String.Encoding.utf8)
                let graphDef = try Tensorflow_GraphDef(textFormatString: text)
                return TFGraph(graphDef: graphDef)
            }
        } catch {
            fatalError("Could not load graph from file " + file.absoluteString)
        }
    }

}
