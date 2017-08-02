//
//  UIViewController.swift
//  Example
//
//  Created by Mathias Claassen on 5/31/17.
//
//

import Bender
import Metal
import UIKit

protocol ExampleViewController {

    var inputSize: LayerSize { get }
    var pixelBufferPool: CVPixelBufferPool? { get set }

}

extension ExampleViewController {

    mutating func setPixelBufferPool() {
        let bufferAttributes = [kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: Int32(kCVPixelFormatType_64RGBAHalf)),
                                kCVPixelBufferWidthKey as String: inputSize.w,
                                kCVPixelBufferHeightKey as String: inputSize.w] as [String: Any]
        CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                [kCVPixelBufferPoolMinimumBufferCountKey as String: 1] as CFDictionary,
                                bufferAttributes as CFDictionary,
                                &pixelBufferPool)
    }

    func getPixelBuffer(from texture: MTLTexture, bufferPool: CVPixelBufferPool) -> CVPixelBuffer? {
        let channels = texture.arrayLength * 4

        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, bufferPool, &pixelBuffer)

        guard let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        if let pointer = CVPixelBufferGetBaseAddress(buffer) {
            let region = MTLRegionMake2D(0, 0, inputSize.w, inputSize.h)
            texture.getBytes(pointer, bytesPerRow: 2 * channels * inputSize.w, from: region, mipmapLevel: 0)
        }
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))

        return buffer
    }

}
