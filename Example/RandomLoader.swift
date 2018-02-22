//
//  RandomLoader.swift
//  Example
//
//  Created by Mathias Claassen on 5/24/17.
//
//

import MetalBender

class RandomParameterLoader: ParameterLoader {

    private(set) var pointer: UnsafeMutablePointer<Float>
    var checkpoint: String = ""

    init(maxSize: Int) {
        let p = malloc(maxSize * MemoryLayout<Float>.stride)
        pointer = p!.bindMemory(to: Float.self, capacity: maxSize)
        uniformRandom(pointer, count: maxSize, scale: 0.1)
    }

    deinit {
        free(pointer)
    }

    func random() -> Float {
        return Float(Double(arc4random()) / Double(UINT32_MAX))
    }

    func uniformRandom(_ x: UnsafeMutablePointer<Float>, count: Int, scale: Float) {
        for i in 0..<count {
            x[i] = (random()*2 - 1) * scale
        }
    }

    func loadWeights(for id: String, modifier: String, size: Int) -> UnsafePointer<Float> {
        return UnsafePointer(pointer)
    }

}
