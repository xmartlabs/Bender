//
//  NetworkLayerProtocol.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import AVFoundation

protocol NetworkLayer {

    var descriptor: MPSImageDescriptor? { get }
    var outputLayerSize: LayerSize { get }
    func initialize(device: MTLDevice, prevSize: LayerSize)
    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage
    func updateCheckpoint(new: String, old: String, device: MTLDevice)
    
}

extension NetworkLayer {

    func createImage(device: MTLDevice) -> MPSImage {
        return MPSImage(device: device, imageDescriptor: descriptor!)
    }

    func createTempImage(buffer: MTLCommandBuffer, descriptor: MPSImageDescriptor? = nil) -> MPSTemporaryImage {
        return MPSTemporaryImage(commandBuffer: buffer, imageDescriptor: descriptor ?? self.descriptor!)
    }

}
