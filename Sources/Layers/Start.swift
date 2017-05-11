//
//  Start.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShaders

public class Start: NetworkLayer {

    public var inputImage: MPSImage!

    init(size: LayerSize) {
        super.init(id: "Palladium_Start")
        outputSize = size
    }

    public override func execute(commandBuffer: MTLCommandBuffer) {
        outputImage = inputImage
    }
    
}
