//
//  CropAndFormatLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

open class Crop: NetworkLayer {
    
    public init(device: MTLDevice, croppedSize: LayerSize, id: String? = nil) {
        super.init(id: id)
        self.outputSize = croppedSize
        self.outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: croppedSize))
    }
    
    open override func execute(commandBuffer: MTLCommandBuffer) {
        let input = getIncoming()
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder.copy(from: input[0].outputImage.texture, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: 0,
                                                 y: (input[0].outputImage.texture.height - outputSize.h) / 2,
                                                 z: 0),
                         sourceSize: MTLSizeMake(outputSize.w, outputSize.h, 1),
                         to: outputImage.texture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitEncoder.endEncoding()
    }
}
