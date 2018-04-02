//
//  Start.swift
//  Bender
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// This layer is used as the starting point for any network. If the inputImage does not have the requested size then it will be resized.
open class Start: NetworkLayer {

    open var inputImage: MPSImage!
    open var lanczos: MPSImageLanczosScale!
    var inputName: String

    public init(size: LayerSize, inputName: String = "") {
        self.inputName = inputName
        super.init(id: "Start_" + inputName)
        outputSize = size
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        lanczos = MPSImageLanczosScale(device: device)
        createOutputs(size: outputSize)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        if inputImage.size == outputSize {
            outputs[executionIndex] = inputImage
            return
        }

        // If the inputImage does not have the requested output size then we have to resize it.
        let inputAspect = Double(inputImage.width) / Double(inputImage.height)
        let aspect = inputAspect - (Double(outputSize.w) / Double(outputSize.h))
        var scaledW: Int
        var scaledH: Int
        if aspect == 0.0 { // input and output aspect ratio are equal
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: outputs[executionIndex].texture)
            return
        } else if aspect > 0.0 { // input aspect ratio greater than output aspect ratio
            scaledH = outputSize.h
            scaledW = Int(Double(outputSize.w) * inputAspect)
        } else { // input aspect ratio smaller than output aspect ratio
            scaledW = outputSize.w
            scaledH = Int(Double(outputSize.h) / inputAspect)
        }

        // SCALE
        let resizedDescriptor = MPSImageDescriptor(channelFormat: .float16,
                                                   width: scaledW,
                                                   height: scaledH,
                                                   featureChannels: inputImage.featureChannels)
        let resizedImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: resizedDescriptor)
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: resizedImg.texture)

        // CROP
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitEncoder.copy(from: resizedImg.texture, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: (scaledW - outputSize.w) / 2,
                                                 y: (scaledH - outputSize.h) / 2,
                                                 z: 0),
                         sourceSize: MTLSizeMake(outputs[executionIndex].width, outputs[executionIndex].height, 1),
                         to: outputs[executionIndex].texture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitEncoder.endEncoding()

        resizedImg.readCount = 0
    }

}
