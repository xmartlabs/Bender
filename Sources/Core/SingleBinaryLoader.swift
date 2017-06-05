//
//  SingleBinaryLoader.swift
//  Bender
//
//  Created by Mathias Claassen on 5/10/17.
//
//

import Foundation

/** Parameter loader that expects all the parameters in one big file. You can either specify the offset for each layer or, by default, it will keep 
    a counter and advacne it with each call to loadWeights(for:modifier:size:)
 */
public class SingleBinaryLoader: ParameterLoader {

    public var checkpoint: String

    private var file: UnsafePointer<Float>?
    private var currentOffset: Int = 0

    /// Maps each layer to the offset of its weight in the file
    public var offsets = [String: Int]()

    /// Defines how the identifier for each pair (id, modifier) is built
    public var offsetString: (_ id: String, _ modifier: String) -> String = { id, modifier in
        return id + modifier
    }

    public init(checkpoint: String) {
        self.checkpoint = checkpoint
    }

    /// Loads the weight file into memory. Should be done before the network is initialized. You must call `reset` after the network is initialized
    ///
    /// - Parameters:
    ///   - count: count of floats in file
    public func loadFile(count: Int) {
        file = load(from: checkpoint, size: count, ofType: "")
    }

    public func loadWeights(for id: String, modifier: String, size: Int) -> UnsafePointer<Float> {
        if let offset = offsets[offsetString(id, modifier)] {
            return file! + offset
        } else {
            let ret = file! + currentOffset
            currentOffset += size
            return ret
        }
    }

    /// Resets counter and weight file. Must be called after network is initialized
    public func reset() {
        currentOffset = 0
        file = nil
    }
    
}
