//
//  Neuron.swift
//  Bender
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Implements the different activation neurons like ReLu, Tanh, Sigmoid, Linear
open class Neuron: NetworkLayer {

    public var type: ActivationNeuronType
    public var neuron: MPSCNNNeuron!

    public init(type: ActivationNeuronType, id: String? = nil) {
        switch type {
        case .none:
            assertionFailure("Cannot create empty neuron layer")
        default:
            break
        }
        self.type = type
        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "Neuron must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        outputSize = incoming[0].outputSize

        self.neuron = type.createNeuron(device: device)
        createOutputs(size: outputSize)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        neuron.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputs[executionIndex], destinationImage: outputs[executionIndex])
    }
}
