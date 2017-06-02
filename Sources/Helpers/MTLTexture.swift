//
//  MTLTexture.swift
//  Bender
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import MetalKit

public extension MTLTexture {

    /// Returns the minimum necessary amount of threadGroups needed to cover a texture given a certain threadGroup size
    func threadGrid(threadGroup: MTLSize) -> MTLSize {
        return MTLSizeMake(max(Int((self.width + threadGroup.width - 1) / threadGroup.width), 1), max(Int((self.height + threadGroup.height - 1) / threadGroup.height), 1), self.arrayLength)
    }

    var size: LayerSize {
        return LayerSize(f: arrayLength * 4, w: width, h: height)
    }

}
