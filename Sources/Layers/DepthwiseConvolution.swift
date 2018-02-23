//
//  DepthwiseConvolution.swift
//  MetalBender
//
//  Created by Mathias Claassen on 2/23/18.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Depthwise Convolution layer
@available(iOS 11.0, *)
open class DepthwiseConvolution: Convolution {

    public required init(convSize: ConvSize, neuronType: ActivationNeuronType = .none, useBias: Bool = false, padding: PaddingType = .same, weights: Data? = nil, bias: Data? = nil, id: String? = nil) {
        super.init(convSize: convSize, neuronType: neuronType, useBias: useBias, padding: padding, weights: weights, bias: bias, id: id)
    }

    open override func getWeightsSize() -> Int {
        return prevSize.f * convSize.kernelHeight * convSize.kernelWidth
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        updateWeights(device: device)
    }

    open override func makeConv(device: MTLDevice, weights: UnsafePointer<Float>, bias: UnsafePointer<Float>?) {
        let desc = MPSCNNDepthWiseConvolutionDescriptor(
            kernelWidth: convSize.kernelWidth,
            kernelHeight: convSize.kernelHeight,
            inputFeatureChannels: prevSize.f,
            outputFeatureChannels: convSize.outputChannels,
            neuronFilter: neuronType.createNeuron(device: device))

        desc.dilationRateX = convSize.dilationX
        desc.dilationRateY = convSize.dilationY
        desc.strideInPixelsX = convSize.strideX
        desc.strideInPixelsY = convSize.strideY

        conv = MPSCNNConvolution(device: device,
                                 convolutionDescriptor: desc,
                                 kernelWeights: weights,
                                 biasTerms: bias,
                                 flags: .none)
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        conv?.encode(commandBuffer: commandBuffer,
                     sourceImage: getIncoming()[0].outputImage,
                     destinationImage: outputImage)
    }

}
