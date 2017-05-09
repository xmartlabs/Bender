//
//  Identity.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShaders

open class Identity: NetworkLayer {

    open override func initialize(device: MTLDevice) {
        outputSize = getIncoming()[0].outputSize
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        outputImage = getIncoming()[0].outputImage
    }

}
