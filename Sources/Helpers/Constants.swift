//
//  Constants.swift
//  VideoStylizer
//
//  Created by Mathias Claassen on 3/14/17.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import CoreVideo
import Metal

struct Constants {

    static let FloatSize = MemoryLayout<Float>.size
    static let HalfSize = MemoryLayout<Float>.size / 2
    static let inputImageWidth = 256
    static let outputImageWidth = 256
//    static let pixelFormat = kCVPixelFormatType_64RGBAHalf
    static let baseKernelCount = 16
//    static let concurrentGPUFrames = 2

}
