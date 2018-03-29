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

    /// If you want to first crop and then scale (for square image outputs). Normally it will first scale and then crop.
    public var useCropScale = false

    public init(size: LayerSize, inputName: String = "") {
        self.inputName = inputName
        super.init(id: "Start_" + inputName)
        outputSize = size
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        lanczos = MPSImageLanczosScale(device: device)
        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        if inputImage.size == outputSize {
            rewireIdentity(at: index, image: inputImage)
            return
        }

        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)

        if outputSize.w == outputSize.h && useCropScale {
            let cropSize = min(inputImage.width, inputImage.height)
            let diff = inputImage.width - inputImage.height
            let cropY = (diff < 0) ? (diff / -2) : 0
            let cropX = (diff < 0) ? 0 : (diff / 2)
            let resizedDescriptor = MPSImageDescriptor(channelFormat: .unorm8, width: cropSize, height: cropSize,
                                                       featureChannels: inputImage.featureChannels)
            // Using MPSImage here as MPSTemporaryImage would sometimes result in rubbish output
            let resizedImg = MPSImage(device: Device.shared, imageDescriptor: resizedDescriptor)
            blitCrop(commandBuffer: commandBuffer, from: inputImage, to: resizedImg, cropX: cropX, cropY: cropY)
            lanczos.encode(commandBuffer: commandBuffer, sourceImage: resizedImg, destinationImage: output)
        } else {
            // If the inputImage does not have the requested output size then we have to resize it.
            let inputAspect = Double(inputImage.width) / Double(inputImage.height)
            let aspect = inputAspect - (Double(outputSize.w) / Double(outputSize.h))
            var scaledW: Int
            var scaledH: Int
            if aspect == 0.0 { // input and output aspect ratio are equal
                lanczos.encode(commandBuffer: commandBuffer,
                               sourceTexture: inputImage.texture,
                               destinationTexture: output.texture)
                return
            } else if aspect > 0.0 { // input aspect ratio greater than output aspect ratio
                scaledH = outputSize.h
                scaledW = Int(Double(outputSize.w) * inputAspect)
            } else { // input aspect ratio smaller than output aspect ratio
                scaledW = outputSize.w
                scaledH = Int(Double(outputSize.h) / inputAspect)
            }

            // SCALE
            let resizedDescriptor = MPSImageDescriptor(channelFormat: .float16, width: scaledW, height: scaledH,
                                                       featureChannels: inputImage.featureChannels)
            let resizedImg = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: resizedDescriptor)
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: resizedImg.texture)

            // CROP
            blitCrop(commandBuffer: commandBuffer, from: resizedImg, to: output,
                     cropX: (scaledW - outputSize.w) / 2, cropY: (scaledH - outputSize.h) / 2)
            resizedImg.readCount = 0
        }
    }

    func blitCrop(commandBuffer: MTLCommandBuffer, from: MPSImage, to: MPSImage, cropX: Int, cropY: Int) {
        let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
        blitEncoder.copy(from: from.texture, sourceSlice: 0, sourceLevel: 0,
                         sourceOrigin: MTLOrigin(x: cropX,
                                                 y: cropY,
                                                 z: 0),
                         sourceSize: MTLSizeMake(to.width, to.height, 1),
                         to: to.texture, destinationSlice: 0, destinationLevel: 0,
                         destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        blitEncoder.endEncoding()
    }

}
