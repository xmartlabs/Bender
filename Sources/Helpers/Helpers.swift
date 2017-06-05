//
//  HelperExtensions.swift
//  Bender
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

/// Function that transposes weights in a certain order
public typealias TransposeFunction = (Data, Shape) -> (Data)

/// Transposes weights from HWIO to OHWI order. Used to pass TensorFlow's weights for Convolution layers
func HWIOtoOHWI(weights: Data, shape: Shape) -> Data {
    var transposed = [Float](repeating: 0.0, count: shape.totalCount)

    for o in 0..<shape.outputChannels {
        for h in 0..<shape.height {
            for w in 0..<shape.width {
                for i in 0..<shape.inputChannels {
                    let tIndex = i + shape.inputChannels * (w + shape.width * (h + shape.height * (o)))
                    let wIndex = o + shape.outputChannels * (i + shape.inputChannels * (w + shape.width * (h)))
                    transposed[tIndex] = weights.pointer()![wIndex]
                }
            }
        }
    }

    return Data.init(bytes: transposed, count: shape.totalCount * MemoryLayout<Float>.stride)
}

func HWIOtoOWHI(weights: Data, shape: Shape) -> Data {
    var transposed = [Float](repeating: 0.0, count: shape.totalCount)

    for o in 0..<shape.outputChannels {
        for h in 0..<shape.height {
            for w in 0..<shape.width {
                for i in 0..<shape.inputChannels {
                    let tIndex = i + shape.inputChannels * (h + shape.height * (w + shape.width * (o)))
                    let wIndex = o + shape.outputChannels * (i + shape.inputChannels * (w + shape.width * (h)))
                    transposed[tIndex] = weights.pointer()![wIndex]
                }
            }
        }
    }

    return Data.init(bytes: transposed, count: shape.totalCount * MemoryLayout<Float>.stride)
}
