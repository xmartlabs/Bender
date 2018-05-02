//
//  ConvolutionLayer.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// 2D Convolution layer
open class Convolution: NetworkLayer {

    /// Used to determine the filename for this layers weights. (Ignored if there is no ParameterLoader)
    public static var weightModifier: String = ""

    /// Used to determine the filename for this layers bias. (Ignored if there is no ParameterLoader)
    public static var biasModifier: String = "bias"

    var weightsPointer: Data?
    var biasPointer: Data?
    var dataSource: Any?
    var cnnDescriptor: MPSCNNConvolutionDescriptor!

    var prevSize: LayerSize!
    public var convSize: ConvSize

    var conv: MPSCNNConvolution?
    let neuronType: ActivationNeuronType
    public var padding: PaddingType

    var useBias: Bool

    public init(convSize: ConvSize, neuronType: ActivationNeuronType = .none, useBias: Bool = false,
                padding: PaddingType = .same, weights: Data? = nil, bias: Data? = nil, id: String? = nil) {
        self.convSize = convSize
        self.neuronType = neuronType
        self.useBias = useBias
        self.padding = padding
        self.weightsPointer = weights
        self.biasPointer = bias
        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "Convolution must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        prevSize = incoming[0].outputSize
        outputSize = LayerSize(h: padding == .same ? prevSize.h / convSize.strideY : (prevSize.h - convSize.kernelHeight) / convSize.strideY + 1,
                               w: padding == .same ? prevSize.w / convSize.strideX : (prevSize.w - convSize.kernelWidth) / convSize.strideX + 1,
                               f: convSize.outputChannels)
        createCNNDescriptor(device: device)
        updateWeights(device: device)
        if padding == .same {
            let padHeight = ((outputSize.h - 1) * convSize.strideY + convSize.kernelHeight - prevSize.h)
            let padWidth  = ((outputSize.w - 1) * convSize.strideX + convSize.kernelWidth - prevSize.w)
            let padTop = Int(padHeight / 2)
            let padLeft = Int(padWidth / 2)

            conv?.offset = MPSOffset(x: ((Int(convSize.kernelWidth)/2) - padLeft), y: (Int(convSize.kernelHeight/2) - padTop), z: 0)
        } else {
            conv?.offset = MPSOffset(x: Int(convSize.kernelWidth)/2, y: Int(convSize.kernelHeight)/2, z: 0)
        }

        createOutputs(size: outputSize, temporary: temporaryImage)
    }

    open func createCNNDescriptor(device: MTLDevice) {
        cnnDescriptor = MPSCNNConvolutionDescriptor(kernelWidth: convSize.kernelWidth,
                                                    kernelHeight: convSize.kernelHeight,
                                                    inputFeatureChannels: prevSize.f,
                                                    outputFeatureChannels: convSize.outputChannels,
                                                    neuronFilter: neuronType.createNeuron(device: device))
        cnnDescriptor.strideInPixelsX = convSize.strideX
        cnnDescriptor.strideInPixelsY = convSize.strideY
        if #available(iOS 11.0, *) {
            cnnDescriptor.dilationRateX = convSize.dilationX
            cnnDescriptor.dilationRateY = convSize.dilationY
        }
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

        if #available(iOS 11.0, *) {
            if let weightsPointer = weightsPointer {
                dataSource = ConvolutionDataSource(cnnDescriptor: cnnDescriptor,
                                                   weights: UnsafeMutableRawPointer(mutating: weightsPointer.pointer()),
                                                   bias: useBias ? UnsafeMutablePointer(mutating: biasPointer?.pointer() as UnsafePointer<Float>?)
                                                    : nil)
            } else {
                dataSource = ConvolutionDataSource(cnnDescriptor: cnnDescriptor, parameterLoader: network.parameterLoader,
                                                   layerId: id, weightCount: getWeightsSize(), biasCount: useBias ? convSize.outputChannels : 0)
            }
            makeConv(device: device, weights: nil, bias: nil)
        } else {
            let weights = weightsPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id,
                                                                                           modifier: Convolution.weightModifier,
                                                                                           size: getWeightsSize())

            var bias: UnsafePointer<Float>? = nil
            if useBias {
                bias = biasPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id,
                                                                                     modifier: Convolution.biasModifier,
                                                                                     size: convSize.outputChannels)
            }
            makeConv(device: device, weights: weights, bias: bias)
        }
    }

    open func makeConv(device: MTLDevice, weights: UnsafePointer<Float>?, bias: UnsafePointer<Float>?) {
        if #available(iOS 11.0, *) {
            // swiftlint:disable:next force_cast
            conv = MPSCNNConvolution(device: device, weights: dataSource as! MPSCNNConvolutionDataSource)
        } else {
            conv = MPSCNNConvolution(device: device,
                                     convolutionDescriptor: cnnDescriptor,
                                     kernelWeights: weights!,
                                     biasTerms: bias,
                                     flags: MPSCNNConvolutionFlags.none)
        }
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {
        conv?.encode(commandBuffer: commandBuffer,
                     sourceImage: getIncoming()[0].getOutput(index: index),
                     destinationImage: getOrCreateOutput(commandBuffer: commandBuffer, index: index))
    }

}
