//
//  LayerSize.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import Foundation

public struct LayerSize {
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

public struct ConvSize {
    public let outputChannels: Int
    public let kernelSize: Int
    public let stride: Int

    public init(outputChannels: Int, kernelSize: Int, stride: Int) {
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
    }
}
