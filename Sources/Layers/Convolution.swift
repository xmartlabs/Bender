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

    private var prevSize: LayerSize!
    public var convSize: ConvSize

    var conv: SlimMPSCNNConvolution?
    let neuronType: ActivationNeuronType
    public var padding: PaddingType

    var useBias: Bool

    public init(convSize: ConvSize, neuronType: ActivationNeuronType = .relu, useBias: Bool = false, padding: PaddingType = .same, id: String? = nil) {
        self.convSize = convSize
        self.neuronType = neuronType
        self.useBias = useBias
        self.padding = padding
        super.init(id: id)
    }
    
    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "Convolution must have one input, not \(incoming.count)")
        prevSize = incoming[0].outputSize
        outputSize = LayerSize(f: convSize.outputChannels,
                               w: padding == .same ? prevSize.w / convSize.stride : (prevSize.w - convSize.kernelSize) / convSize.stride + 1)

        updateWeights(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open func getWeightsSize() -> Int {
        //TODO: not square kernels
        return prevSize.f * convSize.kernelSize * convSize.kernelSize * convSize.outputChannels
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        updateWeights(device: device)
    }

    open func updateWeights(device: MTLDevice) {
        guard let network = network else {
            return
        }

        let weights = network.parameterLoader.loadWeights(for: id, modifier: Convolution.weightModifier, size: getWeightsSize())
        var bias: UnsafePointer<Float>? = nil
        if useBias {
            bias = network.parameterLoader.loadWeights(for: id, modifier: Convolution.biasModifier, size: convSize.outputChannels)
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
        conv?.encode(commandBuffer: commandBuffer,
                     sourceImage: getIncoming()[0].outputImage,
                     destinationImage: outputImage,
                     padding: padding == .same)
    }

}
