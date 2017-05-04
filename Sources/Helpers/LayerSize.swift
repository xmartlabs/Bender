//
//  LayerSize.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import Foundation

struct LayerSize {
    let f: Int
    let w: Int
    let h: Int

    init(f: Int, w: Int) {
        self.f = f
        self.w = w
        self.h = w
    }
    init(f: Int, w: Int, h: Int) {
        self.f = f
        self.w = w
        self.h = h
    }
}

struct ConvSize {
    let outputChannels: Int
    let kernelSize: Int
    let stride: Int
}
