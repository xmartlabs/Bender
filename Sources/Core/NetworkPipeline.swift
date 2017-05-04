//
//  NetworkPipeline.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 11/25/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class Pipeline: NetworkItem {

    fileprivate var layers: [NetworkItem]
    var outputSize: LayerSize!
    
    init(layers: [NetworkLayer]) {
        self.layers = layers
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        var prevSize = prevSize
        for layer in layers {
            layer.initialize(device: device, prevSize: prevSize)
            prevSize = layer.outputSize
        }
        outputSize = prevSize
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {
        for layer in layers {
            layer.updateCheckpoint(new: new, old: old, device: device)
        }
    }

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage {
        var image = inputImage
        for layer in layers {
            image = layer.execute(commandBuffer: commandBuffer,
                                  inputImage: image)
        }
        return image
    }

}

extension Pipeline: Collection, MutableCollection {

    public subscript(position: Int) -> NetworkItem {
        get { return layers[position] }
        set { layers[position] = newValue }
    }

    func index(after i: (Int)) -> (Int) {
        return i + 1
    }

    var startIndex: Int {
        return 0
    }

    var endIndex: Int {
        return layers.count
    }
    
}
