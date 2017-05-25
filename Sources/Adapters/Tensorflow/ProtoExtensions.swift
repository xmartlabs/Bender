//
//  ProtoExtensions.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/19/17.
//
//

import Foundation

extension Tensorflow_NodeDef {

    var shape: Tensorflow_TensorShapeProto? {
        return attr["shape"]?.shape ?? attr["value"]?.tensor.tensorShape
    }

    var strides: (x: Int, y: Int)? {
        guard let strides = attr["strides"]?.list.i,
            let dataFormat = attr["data_format"]?.s,
            let formatString = String(data: dataFormat, encoding: .utf8) else {
                return nil
        }

        let strideX = formatString == "NHWC" ? strides[2] : strides[3]
        let strideY = formatString == "NHWC" ? strides[1] : strides[2]
        return (Int(strideX), Int(strideY))
    }

    var ksize: (width: Int, height: Int)? {
        guard let size = attr["ksize"]?.list.i,
            let dataFormat = attr["data_format"]?.s,
            let formatString = String(data: dataFormat, encoding: .utf8) else {
                return nil
        }

        let width = formatString == "NHWC" ? size[2] : size[3]
        let height = formatString == "NHWC" ? size[1] : size[2]
        return (Int(width), Int(height))
    }

    func activationNeuron() -> ActivationNeuronType {
        var neuron = ActivationNeuronType.none
        if let neuronOp = attr[Constants.CustomAttr.neuron]?.s, let opString = String(data: neuronOp, encoding: .utf8) {
            switch opString {
            case Constants.Ops.Relu:
                neuron = .relu
            case Constants.Ops.Tanh:
                neuron = .tanh
            case Constants.Ops.Sigmoid:
                neuron = .sigmoid
            default:
                break
            }
        }
        return neuron
    }

    func valueData() -> Data? {
        if isTFConstOp, let data = attr["value"]?.tensor.tensorContent {
            return data
        }
        return nil
    }

}

extension Tensorflow_TensorShapeProto {

    var isBias: Bool {
        return dim.count == 1
    }

    var kernelHeight: Int {
        return Int(dim[0].size)
    }

    var kernelWidth: Int {
        return Int(dim[1].size)
    }

    var inputChannels: Int {
        return Int(dim[2].size)
    }

    var outputChannels: Int {
        return Int(dim[3].size)
    }

    var totalCount: Int {
        return outputChannels * inputChannels * kernelWidth * kernelHeight
    }

    // returns the dimensions in WHNC format (unused)
    func orderedValues(format: String) -> [Int] {
        guard format.characters.count == 4 else {
            fatalError("Invalid Format")
        }
        var values = [1, 1, 1, 1]

        for (index, chr) in format.characters.enumerated() {
            let size = Int(self.dim[index].size)
            switch chr {
            case "W":
                values[0] = size
            case "H":
                values[1] = size
            case "N":
                values[2] = size
            case "C":
                values[3] = size
            default:
                break
            }
        }
        
        return values
    }
    
}
