//
//  ConvolutionLayer.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShadersProxy

/// 2D Convolution layer
open class Convolution: NetworkLayer {

    /// Used to determine the filename for this layers weights. (Ignored if there is no ParameterLoader)
    public static var weightModifier: String = ""

    /// Used to determine the filename for this layers bias. (Ignored if there is no ParameterLoader)
    public static var biasModifier: String = "bias"

    var weightsPointer: Data?
    var biasPointer: Data?

    private var prevSize: LayerSize!
    public var convSize: ConvSize

    var conv: MPSCNNConvolution?
    let neuronType: ActivationNeuronType
    public var padding: PaddingType

    var useBias: Bool

    public init(convSize: ConvSize, neuronType: ActivationNeuronType = .none, useBias: Bool = false, padding: PaddingType = .same, weights: Data? = nil, bias: Data? = nil, id: String? = nil) {
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
        if(padding == .same){
            let padHeight = ((outputSize.h - 1) * convSize.strideY + convSize.kernelHeight - prevSize.h)
            let padWidth  = ((outputSize.w - 1) * convSize.strideX + convSize.kernelWidth - prevSize.w)
            let padTop = Int(padHeight / 2)
            let padLeft = Int(padWidth / 2)

            conv?.offset = MPSOffset(x: ((Int(convSize.kernelWidth)/2) - padLeft), y: (Int(convSize.kernelHeight/2) - padTop), z: 0)
        }
        else{
            conv?.offset = MPSOffset(x: Int(convSize.kernelWidth)/2, y: Int(convSize.kernelHeight)/2, z: 0)
        }

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

        let weights = weightsPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id, modifier: Convolution.weightModifier, size: getWeightsSize())
        var bias: UnsafePointer<Float>? = nil
        if useBias {
            bias = biasPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id, modifier: Convolution.biasModifier, size: convSize.outputChannels)
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
