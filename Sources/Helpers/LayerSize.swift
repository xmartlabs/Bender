//
//  LayerSize.swift
//  Bender
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import Foundation

/// Represents an axis within a LayerSize
public enum LayerSizeAxis {
    case f
    case w
    case h

    static var all: [LayerSizeAxis] { return [.f, .w, .h] }

    static func fromTF(index: Int) -> LayerSizeAxis? {
        switch index {
        case 1:
            return .w
        case 2:
            return .h
        case 3:
            return .f
        default:
            return nil
        }
    }

}

/// Represents the size of a NetworkLayer
public struct LayerSize: Equatable {
    public let f: Int
    public let w: Int
    public let h: Int

    public init(h: Int, w: Int, f: Int) {
        self.h = h
        self.w = w
        self.f = f
    }

    subscript(axis: LayerSizeAxis) -> Int {
        switch axis {
        case .f: return f
        case .w: return w
        case .h: return h
        }
    }

    public static func ==(left: LayerSize, right: LayerSize) -> Bool {
        return left.f == right.f && left.w == right.w && left.h == right.h
    }
}


/// Holds variables that determine the size of a Convolution layer
public struct ConvSize {
    public let outputChannels: Int
    public let kernelWidth: Int
    public let kernelHeight: Int
    public let strideX: Int
    public let strideY: Int

    public init(outputChannels: Int, kernelWidth: Int, kernelHeight: Int, strideX: Int, strideY: Int) {
        self.outputChannels = outputChannels
        self.kernelWidth = kernelWidth
        self.kernelHeight = kernelHeight
        self.strideX = strideX
        self.strideY = strideY
    }

    public init(outputChannels: Int, kernelSize: Int, stride: Int) {
        self.init(outputChannels: outputChannels, kernelWidth: kernelSize, kernelHeight: kernelSize, strideX: stride, strideY: stride)
    }

    init(shape: Tensorflow_TensorShapeProto, strideX: Int, strideY: Int) {
        self.init(outputChannels: shape.outputChannels, kernelWidth: shape.kernelWidth, kernelHeight: shape.kernelHeight,
                  strideX: strideX, strideY: strideY)
    }

}
