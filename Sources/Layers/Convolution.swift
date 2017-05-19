//
//  ConvolutionLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

open class Convolution: NetworkLayer {

    static var weightModifier: String = ""
    static var biasModifier: String = "bias"

    var weightsPointer: UnsafePointer<Float>?
    var biasPointer: UnsafePointer<Float>?

    private var prevSize: LayerSize!
    public var convSize: ConvSize

    var conv: SlimMPSCNNConvolution?
    let neuronType: ActivationNeuronType
    public var padding: PaddingType

    var useBias: Bool

    public init(convSize: ConvSize, neuronType: ActivationNeuronType = .relu, useBias: Bool = false, padding: PaddingType = .same, weights: UnsafePointer<Float>? = nil, bias: UnsafePointer<Float>? = nil, id: String? = nil) {
        self.convSize = convSize
        self.neuronType = neuronType
        self.useBias = useBias
        self.padding = padding
        self.weightsPointer = weights
        self.biasPointer = bias
        super.init(id: id)
    }
    
    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "Convolution must have one input, not \(incoming.count)")
        prevSize = incoming[0].outputSize
        outputSize = LayerSize(f: convSize.outputChannels,
                               w: padding == .same ? prevSize.w / convSize.strideX : (prevSize.w - convSize.kernelWidth) / convSize.strideX + 1,
                               h: padding == .same ? prevSize.h / convSize.strideY : (prevSize.h - convSize.kernelHeight) / convSize.strideY + 1)

        updateWeights(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open func getWeightsSize() -> Int {
        return prevSize.f * convSize.kernelHeight * convSize.kernelWidth * convSize.outputChannels
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        updateWeights(device: device)
    }

    open func updateWeights(device: MTLDevice) {
        guard let network = network else {
            return
        }

        let weights = weightsPointer ?? network.parameterLoader.loadWeights(for: id, modifier: Convolution.weightModifier, size: getWeightsSize())
        var bias: UnsafePointer<Float>? = nil
        if useBias {
            bias = biasPointer ?? network.parameterLoader.loadWeights(for: id, modifier: Convolution.biasModifier, size: convSize.outputChannels)
        }

        makeConv(device: device, weights: weights, bias: bias)
    }

    open func makeConv(device: MTLDevice, weights: UnsafePointer<Float>, bias: UnsafePointer<Float>?) {
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: convSize.kernelWidth,
            kernelHeight: convSize.kernelHeight,
            inputFeatureChannels: prevSize.f,
            outputFeatureChannels: convSize.outputChannels,
            neuronFilter: neuronType.createNeuron(device: device))

        desc.strideInPixelsX = convSize.strideX
        desc.strideInPixelsY = convSize.strideY

        conv = SlimMPSCNNConvolution(device: device,
                                     convolutionDescriptor: desc,
                                     kernelWeights: weights,
                                     biasTerms: bias,
                                     flags: .none)
    }
    
    open override func execute(commandBuffer: MTLCommandBuffer) {
        conv?.encode(commandBuffer: commandBuffer,
                     sourceImage: getIncoming()[0].outputImage,
                     destinationImage: outputImage,
                     padding: padding == .same)
    }

}
