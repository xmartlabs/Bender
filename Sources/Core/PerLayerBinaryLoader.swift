//
//  PerLayerBinaryLoader.swift
//  Bender
//
//  Created by Mathias Claassen on 5/10/17.
//
//

import Foundation

/// Parameter loader that expects the parameters separated in different files for each layer
public class PerLayerBinaryLoader: ParameterLoader {

    /// Returns the filename for a specified id and type of parameter
    public var filenameForLayer: ((_ id: String, _ modifier: String) -> String) = { id, modifier in
        return id + modifier
    }

    /// Prefix added to the filenames
    public var checkpoint: String

    /// Suffix added to the filenames. Can be used if the files have an extension for example
    public var suffix: String = ""

    public init(checkpoint: String = "") {
        self.checkpoint = checkpoint
    }

    public func loadWeights(for id: String, modifier: String, size: Int) -> UnsafePointer<Float> {
        let filename = checkpoint + filenameForLayer(id, modifier) + suffix
        return load(from: filename, size: size)
    }
    
}
