//
//  FullyConnected.swift
//  Bender
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShadersProxy

/// FullyConnected or Dense layer.
open class FullyConnected: NetworkLayer {

    /// Used to determine the filename for this layers weights. (Ignored if there is no ParameterLoader)
    public static var weightModifier: String = ""
    /// Used to determine the filename for this layers bias. (Ignored if there is no ParameterLoader)
    public static var biasModifier: String = "bias"

    var weightsPointer: Data?
    var biasPointer: Data?

    var prevSize: LayerSize!
    var neurons: Int

    var kernel: MPSCNNFullyConnected?
    let neuronType: ActivationNeuronType

    var useBias: Bool
    var transpose: TransposeFunction?

    public init(neurons: Int, neuronType: ActivationNeuronType = .none, useBias: Bool = false, weights: Data? = nil, bias: Data? = nil,
                transpose: TransposeFunction? = nil, id: String? = nil) {
        self.neurons = neurons
        self.neuronType = neuronType
        self.useBias = useBias
        self.weightsPointer = weights
        self.biasPointer = bias
        self.transpose = transpose
        super.init(id: id)
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "Fully Connected must have one input, not \(incoming.count)")
        prevSize = incoming.first?.outputSize
        outputSize = LayerSize(f: neurons,
                               w: 1)

        if let transpose = transpose, let weights = weightsPointer {
            weightsPointer = transpose(weights, Shape(width: prevSize.w, height: prevSize.h, inputChannels: prevSize.f, outputChannels: neurons))
        }

        updateWeights(device: device)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open func getWeightsSize() -> Int {
        return prevSize.f * prevSize.w * prevSize.h * neurons
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
            bias = biasPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id, modifier: Convolution.biasModifier, size: neurons)
        }

        makeConv(device: device, weights: weights, bias: bias)
    }

    open func makeConv(device: MTLDevice, weights: UnsafePointer<Float>, bias: UnsafePointer<Float>?) {
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: prevSize.w,
            kernelHeight: prevSize.h,
            inputFeatureChannels: prevSize.f,
            outputFeatureChannels: neurons,
            neuronFilter: neuronType.createNeuron(device: device))

        kernel = MPSCNNFullyConnected(device: device,
                                      convolutionDescriptor: desc,
                                      kernelWeights: weights,
                                      biasTerms: bias,
                                      flags: .none)
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        kernel?.encode(commandBuffer: commandBuffer,
                       sourceImage: getIncoming()[0].outputImage,
                       destinationImage: outputImage)
    }
    
}
