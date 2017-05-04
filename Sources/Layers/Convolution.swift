//
//  ConvolutionLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

enum ActivationNeuronType {
    case relu
    case tanh
    case custom(neuron: MPSCNNNeuron)
    case none

    func createNeuron(device: MTLDevice) -> MPSCNNNeuron? {
        switch self {
        case .relu:
            return MPSCNNNeuronReLU(device: device, a: 0)
        case .tanh:
            return MPSCNNNeuronTanH(device: device, a: 1, b: 1)
        case let .custom(neuron):
            return neuron
        case .none:
            return nil
        }
    }
}

public class Convolution: NetworkLayer {

    private var prevSize: LayerSize!
    var outputSize: LayerSize!
    var convSize: ConvSize

    var padding: Bool
    var conv: SlimMPSCNNConvolution?
    
    var weightsFile: String
    var biasFile: String?
    
    let neuronType: ActivationNeuronType
    
    var outputImage: MPSImage!
    
    init(convSize: ConvSize, neuronType: ActivationNeuronType = .relu, weightsFile: String, biasFile: String? = nil, padding: Bool = true) {
        self.convSize = convSize
        self.neuronType = neuronType
        self.weightsFile = weightsFile
        self.biasFile = biasFile
        self.padding = padding
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {
        self.prevSize = prevSize
        outputSize = LayerSize(f: convSize.outputChannels,
                               w: padding ? prevSize.w / convSize.stride : (prevSize.w - convSize.kernelSize) / convSize.stride + 1)
        updateWeights(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    func getWeightsSize() -> Int {
        return prevSize.f * convSize.kernelSize * convSize.kernelSize * convSize.outputChannels
    }

    func updateCheckpoint(new checkpoint: String, old: String, device: MTLDevice) {
        weightsFile = weightsFile.replacingOccurrences(of: old, with: checkpoint, options: String.CompareOptions.anchored)
        biasFile = biasFile?.replacingOccurrences(of: old, with: checkpoint, options: String.CompareOptions.anchored)
        updateWeights(device: device)
    }

    func updateWeights(device: MTLDevice) {
        let weights = loadWeights(from: weightsFile, size: getWeightsSize())
        var bias: UnsafePointer<Float>? = nil
        if let biasFile = biasFile {
            bias = loadWeights(from: biasFile, size: convSize.outputChannels)
        }

        makeConv(device: device, weights: weights, bias: bias)
    }

    func makeConv(device: MTLDevice, weights: UnsafePointer<Float>, bias: UnsafePointer<Float>?) {
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: convSize.kernelSize,
            kernelHeight: convSize.kernelSize,
            inputFeatureChannels: prevSize.f,
            outputFeatureChannels: convSize.outputChannels,
            neuronFilter: neuronType.createNeuron(device: device))

        desc.strideInPixelsX = convSize.stride
        desc.strideInPixelsY = convSize.stride

        conv = SlimMPSCNNConvolution(device: device,
                                     convolutionDescriptor: desc,
                                     kernelWeights: weights,
                                     biasTerms: bias,
                                     flags: .none)
    }
    
    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage {
        conv?.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage, padding: padding)
        return outputImage
    }

}
