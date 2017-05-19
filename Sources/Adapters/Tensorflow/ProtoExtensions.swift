//
//  ProtoExtensions.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/19/17.
//
//

import Foundation

extension Tensorflow_NodeDef {

    var shape: Tensorflow_TensorShapeProto? {
        return attr["shape"]?.shape
    }

}

extension Tensorflow_TensorShapeProto {

    var isBias: Bool {
        return dim.filter { $0.size == 1 }.count == 3
    }

    var kernelHeight: Int {
        return Int(dim[0].size)
    }

    var kernelWidth: Int {
        return Int(dim[1].size)
    }

    var inputChannels: Int {
        return Int(dim[2].size)
    }

    var outputChannels: Int {
        return Int(dim[3].size)
    }

    var totalCount: Int {
        return outputChannels * inputChannels * kernelWidth * kernelHeight
    }

    // returns the dimensions in WHNC format (unused)
    func orderedValues(format: String) -> [Int] {
        guard format.characters.count == 4 else {
            fatalError("Invalid Format")
        }
        var values = [1, 1, 1, 1]

        for (index, chr) in format.characters.enumerated() {
            let size = Int(self.dim[index].size)
            switch chr {
            case "W":
                values[0] = size
            case "H":
                values[1] = size
            case "N":
                values[2] = size
            case "C":
                values[3] = size
            default:
                break
            }
        }
        
        return values
    }
    
}
