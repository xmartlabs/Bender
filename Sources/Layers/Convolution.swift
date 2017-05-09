//
//  ConvolutionLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

open class Convolution: NetworkLayer {

    private var prevSize: LayerSize!
    public var convSize: ConvSize

    public var padding: Bool
    var conv: SlimMPSCNNConvolution?

    var weightsFile: String
    var biasFile: String?
    
    let neuronType: ActivationNeuronType

    public init(convSize: ConvSize, neuronType: ActivationNeuronType = .relu, weightsFile: String, biasFile: String? = nil, padding: Bool = true, id: String? = nil) {
        self.convSize = convSize
        self.neuronType = neuronType
        self.weightsFile = weightsFile
        self.biasFile = biasFile
        self.padding = padding
        super.init(id: id)
    }
    
    open override func initialize(device: MTLDevice) {
        let incoming = getIncoming()
        assert(incoming.count == 1, "Convolution must have an input")
        self.prevSize = incoming.first?.outputSize
        outputSize = LayerSize(f: convSize.outputChannels,
                               w: padding ? prevSize.w / convSize.stride : (prevSize.w - convSize.kernelSize) / convSize.stride + 1)
        updateWeights(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open func getWeightsSize() -> Int {
        return prevSize.f * convSize.kernelSize * convSize.kernelSize * convSize.outputChannels
    }

    open override func updateCheckpoint(new checkpoint: String, old: String, device: MTLDevice) {
        weightsFile = weightsFile.replacingOccurrences(of: old, with: checkpoint, options: String.CompareOptions.anchored)
        biasFile = biasFile?.replacingOccurrences(of: old, with: checkpoint, options: String.CompareOptions.anchored)
        updateWeights(device: device)
    }

    open func updateWeights(device: MTLDevice) {
        let weights = loadWeights(from: weightsFile, size: getWeightsSize())
        var bias: UnsafePointer<Float>? = nil
        if let biasFile = biasFile {
            bias = loadWeights(from: biasFile, size: convSize.outputChannels)
        }

        makeConv(device: device, weights: weights, bias: bias)
    }

    open func makeConv(device: MTLDevice, weights: UnsafePointer<Float>, bias: UnsafePointer<Float>?) {
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
    
    open override func execute(commandBuffer: MTLCommandBuffer) {
        conv?.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputImage, destinationImage: outputImage, padding: padding)
    }

}
