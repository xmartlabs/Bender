//
//  CropAndFormatLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

class Crop: NetworkLayer {

    var outputSize: LayerSize!
    
    let lanczos: MPSImageLanczosScale?

    // Custom kernels
//    let pipelineBGRAtoRGBA: MTLComputePipelineState

    // Intermediate images
    let outputImage: MPSImage
    
    init(device: MTLDevice, croppedSize: LayerSize, scale: Bool = true) {
        if scale {
            lanczos = MPSImageLanczosScale(device: device)
        }
        self.outputSize = croppedSize

        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: croppedSize))
//        // Load custom metal kernels
//        do {
//            let library = device.newDefaultLibrary()!
//            let bgra_to_rgba = library.makeFunction(name: "bgra_to_rgba")
//            pipelineBGRAtoRGBA = try device.makeComputePipelineState(function: bgra_to_rgba!)
//        } catch {
//            print(error)
//            fatalError("Error initializing compute pipeline")
//        }
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {}

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}
    
    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage {
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder.copy(from: inputImage.texture, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: 0,
                                                 y: (inputImage.texture.height - outputSize.h) / 2,
                                                 z: 0),
                         sourceSize: MTLSizeMake(outputSize.w, outputSize.h, 1),
                         to: outputImage.texture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitEncoder.endEncoding()

        return outputImage
    }
}
