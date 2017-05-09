//
//  Neuron.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/8/17.
//
//

import MetalPerformanceShaders

open class Neuron: NetworkLayer {

    public var type: ActivationNeuronType
    public var neuron: MPSCNNNeuron!

    init(type: ActivationNeuronType, id: String? = nil) {
        switch type {
        case .none:
            assertionFailure("Cannot create empty neuron layer")
        default:
            break
        }
        self.type = type
        super.init(id: id)
    }

    open override func initialize(device: MTLDevice) {
        outputSize = getIncoming()[0].outputSize

        self.neuron = type.createNeuron(device: device)!
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        neuron.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputImage, destinationImage: outputImage)
    }
}
