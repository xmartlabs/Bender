//
//  TFMappers.swift
//  Bender
//
//  Created by Mathias Claassen on 5/24/17.
//
//

import Foundation

public extension TFConverter {

    func setupMappers() {
        //MARK: Add
        let addMapper = { (node: TFNode) in
            return Add(id: node.nodeDef.name)
        }
        mappers[Constants.Ops.Add] = addMapper

        //MARK: Activation neurons
        let neuronMapper = { (type: ActivationNeuronType) -> (TFNode) -> NetworkLayer in
            { node in
                return Neuron(type: type, id: node.nodeDef.name)
            }
        }

        mappers[Constants.Ops.Relu] = neuronMapper(.relu)
        mappers[Constants.Ops.Tanh] = neuronMapper(.tanh)
        mappers[Constants.Ops.Sigmoid] = neuronMapper(.sigmoid)

        //MARK: Pooling
        let poolingMapper = { (type: PoolingType) -> (TFNode) -> NetworkLayer in { node in
            guard let pad = node.nodeDef.attr["padding"]?.s,
                let padString = String(data: pad, encoding: .utf8),
                let strides = node.nodeDef.strides,
                let size = node.nodeDef.ksize else {
                    fatalError("Cannot create MaxPool")
            }

            return Pooling(type: type,
                           padding: PaddingType.fromTF(padString),
                           kernelSize: (width: Int(size.width), height: Int(size.height)),
                           stride: (x: strides.x, y: strides.y),
                           id: node.nodeDef.name)
            }
        }
        mappers["MaxPool"] = poolingMapper(.max)
        mappers["AvgPool"] = poolingMapper(.avg)

        //MARK: SoftMax
        let softMaxMapper = { (node: TFNode) in
            return Softmax(id: node.nodeDef.name)
        }
        mappers["Softmax"] = softMaxMapper


        //MARK: Conv
        let convMapper = { (node: TFNode) -> NetworkLayer in
            guard let pad = node.nodeDef.attr["padding"]?.s,
                let padString = String(data: pad, encoding: .utf8),
                let strides = node.nodeDef.strides else {
                    fatalError("Cannot create Conv2D")
            }

            guard let weightData = TFWeightData.getWeightData(node: node) else {
                fatalError("Could not get weight information for this Conv2DTranspose")
            }

            //transpose weights
            let weights = weightData.weights != nil ? HWIOtoOHWI(weights: weightData.weights!, shape: weightData.weightShape.toShape) : nil

            let convSize = ConvSize(shape: weightData.weightShape,
                                    strideX: Int(strides.x),
                                    strideY: Int(strides.y))
            return Convolution(convSize: convSize,
                               neuronType: node.nodeDef.activationNeuron(),
                               useBias: weightData.useBias,
                               padding: PaddingType.fromTF(padString),
                               weights: weights,
                               bias: weightData.bias,
                               id: node.nodeDef.name)
        }
        mappers["Conv2D"] = convMapper

        //MARK: ConvTranspose
        let convTransposeMapper = { (node: TFNode) -> NetworkLayer in
            guard let strides = node.nodeDef.strides else {
                fatalError("Cannot create Conv2DTranspose")
            }

            guard let weightData = TFWeightData.getWeightData(node: node) else {
                fatalError("Could not get weight information for this Conv2DTranspose")
            }

            let convSize = ConvSize(outputChannels: Int(weightData.weightShape.dim[2].size), // outputChannels for ConvTranspose are at 3rd dim
                                    kernelWidth: weightData.weightShape.kernelWidth,
                                    kernelHeight: weightData.weightShape.kernelHeight,
                                    strideX: Int(strides.x),
                                    strideY: Int(strides.y))
            return ConvTranspose(size: convSize,
                                 weights: weightData.weights,
                                 bias: weightData.bias,
                                 id: node.nodeDef.name)
        }
        mappers["Conv2DBackpropInput"] = convTransposeMapper

        //MARK: Dense
        let denseMapper = { (node: TFNode) -> NetworkLayer in
            guard let weightData = TFWeightData.getWeightData(node: node) else {
                fatalError("Could not get weight information for this Conv2DTranspose")
            }



            // transpose weights is done in the layer itself

            return FullyConnected(neurons: Int(weightData.weightShape.dim[1].size),
                                  neuronType: node.nodeDef.activationNeuron(),
                                  useBias: weightData.useBias,
                                  weights: weightData.weights,
                                  bias: weightData.bias,
                                  transpose: HWIOtoOWHI,
                                  id: node.nodeDef.name)
        }
        mappers[Constants.Ops.Dense] = denseMapper

        //MARK: InstanceNorm
        let inormMapper = { (node: TFNode) -> NetworkLayer in
            guard let incoming = node.incomingNodes()  as? [TFNode],
                let shiftVar = incoming.first(where: { $0.nodeDef.isTFVariableOrConstOp }),
                let mul = incoming.first(where: { $0.nodeDef.isTFInstanceNormMulOp }),
                let scaleVar = mul.incomingNodes().first as? TFNode, scaleVar.nodeDef.isTFVariableOrConstOp else {
                    fatalError("Could not parse Instance norm node")
            }

            var scale: Data?
            var shift: Data?

            if shiftVar.nodeDef.isTFConstOp {
                shift = shiftVar.nodeDef.valueData()
            } else {
                shift = (shiftVar.incomingNodes().first as? TFNode)?.nodeDef.valueData()
            }

            if scaleVar.nodeDef.isTFConstOp {
                scale = scaleVar.nodeDef.valueData()
            } else {
                scale = (scaleVar.incomingNodes().first as? TFNode)?.nodeDef.valueData()
            }

            return InstanceNorm(scale: scale, shift: shift, id: node.nodeDef.name)
        }
        mappers[Constants.Ops.InstanceNormAdd] = inormMapper
    }
    
}
