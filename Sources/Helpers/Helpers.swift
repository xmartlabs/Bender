//
//  HelperExtensions.swift
//  Palladium
//
//  Created by Joaquin Rocco on 11/30/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import Foundation

/// Helper function to measure the time elapsed during the execution of a block of code
public func measure(_ label: String = "", _ block: () -> ()) {
    let time1 = Date()
    block()
    let time2 = Date()
    let v = time2.timeIntervalSince(time1)
    debugPrint("\(label): \(v) (\(1/v) per second)")
}

/// Transposes weights from HWIO to OHWI order. Used to pass TensorFlow's weights for Convolution layers
func HWIOtoOHWI(weights: Data, shape: Tensorflow_TensorShapeProto) -> Data {
    var transposed = [Float](repeating: 0.0, count: shape.totalCount)

    for o in 0..<shape.outputChannels {
        for h in 0..<shape.kernelHeight {
            for w in 0..<shape.kernelWidth {
                for i in 0..<shape.inputChannels {
                    let tIndex = i + shape.inputChannels * (w + shape.kernelWidth * (h + shape.kernelHeight * (o)))
                    let wIndex = o + shape.outputChannels * (i + shape.inputChannels * (w + shape.kernelWidth * (h)))
                    transposed[tIndex] = weights.pointer()![wIndex]
                }
            }
        }
    }

    return Data.init(bytes: transposed, count: shape.totalCount * MemoryLayout<Float>.stride)
}
