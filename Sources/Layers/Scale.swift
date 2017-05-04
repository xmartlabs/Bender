//
//  LanczosLayer.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 4/24/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

class Scale: NetworkLayer {

    var outputSize: LayerSize!

    var lanczos: MPSImageLanczosScale!

    // Intermediate images
    var outputImage: MPSImage!

    init(layerSize: LayerSize) {
        self.outputSize = layerSize
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        lanczos = MPSImageLanczosScale(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage {
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: outputImage.texture)

        return outputImage
    }
}
