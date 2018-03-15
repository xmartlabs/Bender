//
//  HelperExtensions.swift
//  Bender
//
//  Created by Joaquin Rocco on 11/30/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import Accelerate

/// Helper function to measure the time elapsed during the execution of a block of code
public func measure(_ label: String = "", _ block: () -> Void) {
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
    let weightsPointer: UnsafePointer<Float>! = weights.pointer()

    for o in 0..<shape.outputChannels {
        for h in 0..<shape.height {
            for w in 0..<shape.width {
                for i in 0..<shape.inputChannels {
                    let tIndex = i + shape.inputChannels * (w + shape.width * (h + shape.height * (o)))
                    let wIndex = o + shape.outputChannels * (i + shape.inputChannels * (w + shape.width * (h)))
                    transposed[tIndex] = weightsPointer[wIndex]
                }
            }
        }
    }

    return Data(bytes: transposed, count: shape.totalCount * MemoryLayout<Float>.stride)
}

public func permute(order: [Int]) -> (_ weights: Data, _ shape: Shape) -> Data {
    return { weights, shape in
        var transposed = [Float](repeating: 0.0, count: shape.totalCount)
        let weightsPointer: UnsafePointer<Float>! = weights.pointer()

        var index = 0
        var index_order = [0, 0, 0, 0]
        for d in 0..<shape.at(order[0]) {

            index_order[order[0]] = d
            for c in 0..<shape.at(order[1]) {
                index_order[order[1]] = c
                for b in 0..<shape.at(order[2]) {
                    index_order[order[2]] = b
                    for a in 0..<shape.at(order[3]) {
                        index_order[order[3]] = a
                        let tIndex = index_order[3] + shape.at(3) * (index_order[2] + shape.at(2) * (index_order[1] + shape.at(1) * (index_order[0])))
                        transposed[index] = weightsPointer[tIndex]
                        index += 1
                    }
                }
            }
        }

        return Data(bytes: transposed, count: shape.totalCount * MemoryLayout<Float>.stride)
    }
}

/// Transposes weights from HWIO to OHWI order. Used to pass TensorFlow's weights for Convolution layers
func HWIOtoIOWH(weights: Data, shape: Shape) -> Data {
    var transposed = [Float](repeating: 0.0, count: shape.totalCount)
    let weightsPointer: UnsafePointer<Float>! = weights.pointer()

    for o in 0..<shape.outputChannels {
        for h in 0..<shape.height {
            for w in 0..<shape.width {
                for i in 0..<shape.inputChannels {
                    let tIndex = h + shape.height * (w + shape.width * (o + shape.outputChannels * (i)))
                    let wIndex = o + shape.outputChannels * (i + shape.inputChannels * (w + shape.width * (h)))
                    transposed[tIndex] = weightsPointer[wIndex]
                }
            }
        }
    }

    return Data(bytes: transposed, count: shape.totalCount * MemoryLayout<Float>.stride)
}

public func float32to16(_ input: UnsafeMutablePointer<Float>, count: Int) -> [UInt16] {
    var output = [UInt16](repeating: 0, count: count)
    var bufferFloat32 = vImage_Buffer(data: input, height: 1, width: UInt(count), rowBytes: count * 4)
    var bufferFloat16 = vImage_Buffer(data: &output, height: 1, width: UInt(count), rowBytes: count * 2)

    if vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0) != kvImageNoError {
        fatalError("Error converting float32 to float16")
    }
    return output
}
