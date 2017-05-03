//
//  LanczosLayer.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 4/24/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

class LanczosLayer: NetworkLayer {

    var outputLayerSize: LayerSize {
        return layerSize
    }

    var layerSize: LayerSize!
    var lanczos: MPSImageLanczosScale!

    // Intermediate images
    var descriptor: MPSImageDescriptor?
    var outputImage: MPSImage!

    init(layerSize: LayerSize) {
        self.layerSize = layerSize
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        lanczos = MPSImageLanczosScale(device: device)
        descriptor = MPSImageDescriptor(channelFormat: .float16, width: layerSize.w, height: layerSize.h, featureChannels: layerSize.f)
        outputImage = createImage(device: device)
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: outputImage.texture)

        return outputImage
    }
}
