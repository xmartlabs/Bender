//
//  ActivationNeuronType.swift
//  Bender
//
//  Created by Mathias Claassen on 5/9/17.
//
//

import MetalPerformanceShadersProxy

/// Activation neuron type
public enum ActivationNeuronType {

    case relu
    case tanh
    case scaleFloat
    case sigmoid
    case custom(neuron: MPSCNNNeuron)
    case none

    public func createNeuron(device: MTLDevice) -> MPSCNNNeuron? {
        switch self {
        case .relu:
            return MPSCNNNeuronReLU(device: device, a: 0)
        case .scaleFloat:
            return MPSCNNNeuronLinear(device: device, a: 0.5, b: 0.5)
        case .sigmoid:
            return MPSCNNNeuronSigmoid(device: device)
        case .tanh:
            return MPSCNNNeuronTanH(device: device, a: 1, b: 1)
        case let .custom(neuron):
            return neuron
        case .none:
            return nil
        }
    }
}
