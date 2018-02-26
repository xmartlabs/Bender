//
//  CropAndFormatLayer.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

extension MTLOrigin {
    static let zero = MTLOrigin()
}

/// This layer crops the input image to the desired size. The cropRect is taken from the center of the input image.
open class Crop: NetworkLayer {

    public init(device: MTLDevice, croppedSize: LayerSize, id: String? = nil) {
        super.init(id: id)
        self.outputSize = croppedSize
        self.outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: croppedSize))
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "Crop must have one input, not \(incoming.count)")
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        let input = getIncoming()[0].outputImage!
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitEncoder.copy(from: input.texture, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: (input.width - outputSize.w) / 2,
                                                 y: (input.height - outputSize.h) / 2,
                                                 z: 0),
                         sourceSize: MTLSizeMake(outputSize.w, outputSize.h, 1),
                         to: outputImage.texture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: .zero)
        blitEncoder.endEncoding()
    }
}
