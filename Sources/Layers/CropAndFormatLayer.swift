//
//  CropAndFormatLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

class CropAndFormatLayer: NetworkLayer {

    var outputLayerSize = LayerSize(f: 3, w: 256)
    
    let lanczos: MPSImageLanczosScale
    // Custom kernels
    let pipelineBGRAtoRGBA: MTLComputePipelineState
    
    // Intermediate images
    var descriptor: MPSImageDescriptor?
    let croppedImg: MPSImage
    
    init(device: MTLDevice) {
        lanczos = MPSImageLanczosScale(device: device)
        
        // Load custom metal kernels
        do {
            let library = device.newDefaultLibrary()!
            let bgra_to_rgba = library.makeFunction(name: "bgra_to_rgba")
            pipelineBGRAtoRGBA = try device.makeComputePipelineState(function: bgra_to_rgba!)
        } catch {
            print(error)
            fatalError("Error initializing compute pipeline")
        }
        
        // init intermediate images
        croppedImg = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: outputLayerSize.w, height: outputLayerSize.h, featureChannels: 3))
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {}

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}
    
    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {
        let inputTexture = inputImage.texture.makeTextureView(pixelFormat: .bgra8Unorm)
        
        // Resize -> 256x~341
        let resizedDescriptor = MPSImageDescriptor(channelFormat: .float16, width: outputLayerSize.w, height: inputTexture.height*outputLayerSize.w/inputTexture.width, featureChannels: 3)
        let resizedImg = MPSImage(device: commandBuffer.device, imageDescriptor: resizedDescriptor)
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputTexture, destinationTexture: resizedImg.texture)
        
        // CROP IMAGE -> 256x256
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder.copy(from: resizedImg.texture, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: 0,
                                                 y: (inputTexture.height * outputLayerSize.w / inputTexture.width - outputLayerSize.h) / 2,
                                                 z: 0),
                         sourceSize: MTLSizeMake(croppedImg.width, croppedImg.height, 1),
                         to: croppedImg.texture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitEncoder.endEncoding()

        return croppedImg
    }
}
