//
//  ActivationNeuronType.swift
//  Bender
//
//  Created by Mathias Claassen on 5/9/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Activation neuron type
public enum ActivationNeuronType {

    case custom(neuron: MPSCNNNeuron)
    case relu
    case relu6
    case scaleFloat
    case sigmoid
    case tanh
    case none

    public func createNeuron(device: MTLDevice) -> MPSCNNNeuron? {
        switch self {
        case .relu:
            return MPSCNNNeuronReLU(device: device, a: 0)
        case .relu6:
            if #available(iOS 11.0, *) {
                return MPSCNNNeuronReLUN(device: device, a: 0, b: 6)
            } else {
                // In iOS 10 we return Relu but print warning
                // TODO: Implement for iOS 10
                debugPrint("WARNING: Relu6 not supported in iOS 10")
                return MPSCNNNeuronReLU(device: device, a: 0)
            }
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
