//
//  WatermarkLayer.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 4/24/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import AVFoundation
import MetalKit
import MetalPerformanceShaders

/**
 Applies a watermark to a texture
 **/
class WatermarkLayer: NetworkLayer {
    var descriptor: MPSImageDescriptor?

    var outputLayerSize = LayerSize(f: 3, w: 256)

    // Custom kernels
    let pipelineWatermark: MTLComputePipelineState

    var watermarkTexture: MTLTexture!

    let outputImage : MPSImage

    init(device: MTLDevice) {
        // Load custom metal kernels
        do {
            let library = device.newDefaultLibrary()!
            let kernel = library.makeFunction(name: "apply_watermark")
            pipelineWatermark = try device.makeComputePipelineState(function: kernel!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }

        // init intermediate images
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputLayerSize))
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        let loader = MTKTextureLoader(device: device)
        do {
            let path = Bundle.main.path(forResource: "watermark", ofType: "png")!
            let data = try NSData(contentsOfFile: path) as Data
            watermarkTexture = try loader.newTexture(with: data, options: [MTKTextureLoaderOptionSRGB : (false as NSNumber)])
        } catch {
            fatalError("Could not initialize watermark texture")
        }
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {

        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "Watermark encoder"
        encoder.setComputePipelineState(pipelineWatermark)
        encoder.setTexture(inputImage.texture, at: 0)
        encoder.setTexture(watermarkTexture, at: 1)
        encoder.setTexture(outputImage.texture, at: 2)
        let threadsPerGroups = MTLSizeMake(32, 8, 1)
        let threadGroups = MTLSizeMake(outputImage.texture.width / threadsPerGroups.width,
                                       outputImage.texture.height / threadsPerGroups.height, 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        return outputImage
    }

}
