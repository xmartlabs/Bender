//
//  ParameterLoader.swift
//  Bender
//
//  Created by Mathias Claassen on 5/10/17.
//
//

import Foundation

/// Protocol implemented by the different types of parameter loaders
public protocol ParameterLoader {

    /// Weight checkpoint. This variable is the prefix of the weight files.
    var checkpoint: String { get set }

    /// Loads weights for a single buffer
    ///
    /// - Parameters:
    ///   - id: NetworkLayer id
    ///   - modifier: The type of weights to load (e.g. bias, scale, shift). Use to distinguish the different parameters needed for a layer.
    ///   - size: Amount of floats to load
    /// - Returns: A pointer to the loaded floats
    func loadWeights(for id: String, modifier: String, size: Int) -> UnsafePointer<Float>

}

extension ParameterLoader {

    func load(from file: String, size: Int, ofType: String = "", useFloat16: Bool = false) -> UnsafePointer<Float> {
        // Load weights from file(s)
        let typeSize = useFloat16 ? Constants.HalfSize : Constants.FloatSize
        let sizeWeights = size * typeSize

        // get the url to this layer's weights
        let wtPath = Bundle.main.path( forResource: file, ofType: ofType)

        // open file descriptors in read-only mode to parameter files
        let fd_w  = open( wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)

        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")

        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0);

        // cast Void pointers to Float
        let w = UnsafePointer(hdrW!.bindMemory(to: Float.self, capacity: Int(sizeWeights)))

        close(fd_w)
        assert(w != UnsafePointer<Float>.init(bitPattern: -1), "mmap failed with errno = \(errno)")
        return w
    }

}

class NoParameterLoader: ParameterLoader {

    var checkpoint: String = ""

    init() {}
    
    func loadWeights(for id: String, modifier: String, size: Int) -> UnsafePointer<Float> {
        fatalError("No parameter loader has been specified!")
    }
}
