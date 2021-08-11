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

    public required override init(convSize: ConvSize, neuronType: ActivationNeuronType = .none, useBias: Bool = false,
                                  padding: PaddingType = .same, weights: Data? = nil, bias: Data? = nil,
                                  paddings: [Int] = [0, 0, 0, 0], edgeMode: MPSImageEdgeMode = .zero,
                                  id: String? = nil) {
        super.init(convSize: convSize, neuronType: neuronType, useBias: useBias, padding: padding,
                   weights: weights, bias: bias, paddings: paddings, edgeMode: edgeMode, id: id)
    }

    open override func createCNNDescriptor(device: MTLDevice) {
        cnnDescriptor = MPSCNNDepthWiseConvolutionDescriptor(kernelWidth: convSize.kernelWidth,
                                                             kernelHeight: convSize.kernelHeight,
                                                             inputFeatureChannels: prevSize.f,
                                                             outputFeatureChannels: convSize.outputChannels,
                                                             neuronFilter: neuronType.createNeuron(device: device))
        cnnDescriptor.dilationRateX = convSize.dilationX
        cnnDescriptor.dilationRateY = convSize.dilationY
        cnnDescriptor.strideInPixelsX = convSize.strideX
        cnnDescriptor.strideInPixelsY = convSize.strideY
    }

    open override func getWeightsSize() -> Int {
        return prevSize.f * convSize.kernelHeight * convSize.kernelWidth
    }

}
