//
//  LayerSize.swift
//  Bender
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import Foundation

/// Represents the size of a NetworkLayer
public struct LayerSize: Equatable {
    public let f: Int
    public let w: Int
    public let h: Int

    public init(f: Int, w: Int) {
        self.f = f
        self.w = w
        self.h = w
    }

    public init(f: Int, w: Int, h: Int) {
        self.f = f
        self.w = w
        self.h = h
    }
}

public func == (left: LayerSize, right: LayerSize) -> Bool {
    return left.f == right.f && left.w == right.w && left.h == right.h
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
