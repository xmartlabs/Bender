//
//  HelperExtensions.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 11/30/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import Foundation

public func measure(_ label: String = "", _ block: () -> ()) {
    let time1 = Date()
    block()
    let time2 = Date()
    let v = time2.timeIntervalSince(time1)
    print("\(label): \(v) (\(1/v) per second)")
}

func HWIOtoOHWI(weights: UnsafePointer<Float>, shape: Tensorflow_TensorShapeProto) -> UnsafePointer<Float> {
    var transposed = [Float](repeating: 0.0, count: shape.totalCount)

    for o in 0..<shape.outputChannels {
        for h in 0..<shape.kernelHeight {
            for w in 0..<shape.kernelWidth {
                for i in 0..<shape.inputChannels {
                    let tIndex = i + shape.inputChannels * (w + shape.kernelWidth * (h + shape.kernelHeight * (o)))
                    let wIndex = o + shape.outputChannels * (i + shape.inputChannels * (w + shape.kernelWidth * (h)))
                    transposed[tIndex] = weights[wIndex]
                }
            }
        }
    }

    return UnsafePointer(transposed)
}
