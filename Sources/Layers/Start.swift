//
//  Start.swift
//  Bender
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShadersProxy

/// This layer is used as the starting point for any network. If the inputImage does not have the requested size then it will be resized.
public class Start: NetworkLayer {

    public var inputImage: MPSImage!
    var lanczos: MPSImageLanczosScale!
    var croppedImg: MPSImage!

    init(size: LayerSize) {
        super.init(id: "Bender_Start")
        outputSize = size
    }

    public override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        lanczos = MPSImageLanczosScale(device: device)
        croppedImg = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    public override func execute(commandBuffer: MTLCommandBuffer) {
        if inputImage.size == outputSize {
            outputImage = inputImage
            return
        }

        // If the inputImage does not have the requested output size then we have to resize it.

        outputImage = croppedImg
        let inputAspect = Double(inputImage.width) / Double(inputImage.height)
        let aspect = inputAspect - (Double(outputSize.w) / Double(outputSize.h))
        var scaledW: Int
        var scaledH: Int
        if aspect == 0.0 { // input and output aspect ratio are equal
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: croppedImg.texture)
            return
        } else if aspect > 0.0 { // input aspect ratio greater than output aspect ratio
            scaledH = outputSize.h
            scaledW = Int(Double(outputSize.w) * inputAspect)
        } else { // input aspect ratio smaller than output aspect ratio
            scaledW = outputSize.w
            scaledH = Int(Double(outputSize.h) / inputAspect)
        }

        // SCALE
        let resizedDescriptor = MPSImageDescriptor(channelFormat: .float16, width: scaledW, height: scaledH, featureChannels: inputImage.featureChannels)
        let resizedImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: resizedDescriptor)
        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: resizedImg.texture)

        // CROP
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()
        blitEncoder.copy(from: resizedImg.texture, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: (scaledW - outputSize.w) / 2,
                                                 y: (scaledH - outputSize.h) / 2,
                                                 z: 0),
                         sourceSize: MTLSizeMake(croppedImg.width, croppedImg.height, 1),
                         to: croppedImg.texture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitEncoder.endEncoding()

        resizedImg.readCount = 0
    }

}
