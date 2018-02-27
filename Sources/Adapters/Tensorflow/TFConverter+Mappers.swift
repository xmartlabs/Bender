//
//  TFMappers.swift
//  Bender
//
//  Created by Mathias Claassen on 5/24/17.
//
//

import Foundation

public extension TFConverter {

    private func addMapper(node: TFNode) -> NetworkLayer {
        return Add(id: node.nodeDef.name)
    }

    private func neuronMapper(_ type: ActivationNeuronType) -> (TFNode) -> NetworkLayer {
        return { node in Neuron(type: type, id: node.nodeDef.name) }
    }

    private func softMaxMapper(node: TFNode) -> NetworkLayer {
        return Softmax(id: node.nodeDef.name)
    }

    private func poolingMapper(_ type: PoolingType) -> (TFNode) -> NetworkLayer {
        return { node in
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

    private func convMapper(node: TFNode) -> NetworkLayer {
        guard let pad = node.nodeDef.attr["padding"]?.s,
            let padString = String(data: pad, encoding: .utf8),
            let strides = node.nodeDef.strides else {
                fatalError("Cannot create Conv2D")
        }
        let dilations = node.nodeDef.dilations ?? (1, 1)

        guard let weightData = TFWeightData.getWeightData(node: node) else {
            fatalError("Could not get weight information for this Conv2DTranspose")
        }

        if node.nodeDef.isTFDepthwiseConvOp, #available(iOS 11.0, *) {
            //transpose weights
            let weights = weightData.weights != nil ? HWIOtoIOWH(weights: weightData.weights!, shape: weightData.weightShape.toShape) : nil
            // Depthwise does not change feature channels count
            let convSize = ConvSize(outputChannels: weightData.weightShape.inputChannels,
                                    kernelWidth: weightData.weightShape.kernelHeight,
                                    kernelHeight: weightData.weightShape.kernelWidth,
                                    strideX: Int(strides.x),
                                    strideY: Int(strides.y),
                                    dilationX: dilations.x,
                                    dilationY: dilations.y)
            return DepthwiseConvolution(convSize: convSize,
                                        neuronType: node.nodeDef.activationNeuron(),
                                        useBias: weightData.useBias,
                                        padding: PaddingType.fromTF(padString),
                                        weights: weights,
                                        bias: weightData.bias,
                                        id: node.nodeDef.name)
        } else {
            //transpose weights
            let weights = weightData.weights != nil ? HWIOtoOHWI(weights: weightData.weights!, shape: weightData.weightShape.toShape) : nil
            let convSize = ConvSize(shape: weightData.weightShape,
                                    strideX: Int(strides.x),
                                    strideY: Int(strides.y),
                                    dilationX: dilations.x,
                                    dilationY: dilations.y)
            return Convolution(convSize: convSize,
                               neuronType: node.nodeDef.activationNeuron(),
                               useBias: weightData.useBias,
                               padding: PaddingType.fromTF(padString),
                               weights: weights,
                               bias: weightData.bias,
                               id: node.nodeDef.name)
        }
    }

    private func convTransposeMapper(node: TFNode) -> NetworkLayer {
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

    private func denseMapper(node: TFNode) -> NetworkLayer {
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

    private func inormMapper(node: TFNode) -> NetworkLayer {
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

    private func concatMapper(node: TFNode) -> NetworkLayer {
        let inputs = node.incomingNodes().flatMap { $0 as? TFNode }
        let axisNodeV2 = inputs.first(where: { $0.nodeDef.name == "\(node.nodeDef.name)/axis" })
        let axisNodeV1 = inputs.first(where: { $0.nodeDef.name == "\(node.nodeDef.name)/concat_dim" })
        guard let axisNode = axisNodeV2 ?? axisNodeV1 else {
            fatalError("Cannot create \(Constants.Ops.Concat). Missing input node \(node.nodeDef.name)/axis")
        }
        guard let axisInt32 = axisNode.nodeDef.attr["value"]?.tensor.intVal.first, let axis = LayerSizeAxis.fromTF(index: Int(axisInt32)) else {
            fatalError("Cannot create \(Constants.Ops.Concat). Missing or invalid attribute axis.")
        }
        return Concat(axis: axis, id: node.nodeDef.name)
    }

    private func batchnormMapper(node: TFNode) -> NetworkLayer {
        guard let variables = (node.incomingNodes() as? [TFNode])?.filter({ $0.nodeDef.isTFVariableOrConstOp }),
            let meanVar = variables.first(where: { $0.nodeDef.isTFMovMean }),
            let varianceVar = variables.first(where: { $0.nodeDef.isTFMovVariance }),
            let gammaVar = variables.first(where: { $0.nodeDef.isTFGamma }),
            let betaVar = variables.first(where: { $0.nodeDef.isTFBeta }) else {
                fatalError("Could not parse Batch norm node")
        }

        var mean: Data?
        var variance: Data?
        var scale: Data?
        var offset: Data?

        if meanVar.nodeDef.isTFConstOp {
            mean = meanVar.nodeDef.valueData()
        } else {
            mean = (meanVar.incomingNodes().first as? TFNode)?.nodeDef.valueData()
        }

        if varianceVar.nodeDef.isTFConstOp {
            variance = varianceVar.nodeDef.valueData()
        } else {
            variance = (varianceVar.incomingNodes().first as? TFNode)?.nodeDef.valueData()
        }

        if betaVar.nodeDef.isTFConstOp {
            offset = betaVar.nodeDef.valueData()
        } else {
            offset = (betaVar.incomingNodes().first as? TFNode)?.nodeDef.valueData()
        }

        if node.nodeDef.attr["scale_after_normalization"]?.b == true {
            if gammaVar.nodeDef.isTFConstOp {
                scale = gammaVar.nodeDef.valueData()
            } else {
                scale = (gammaVar.incomingNodes().first as? TFNode)?.nodeDef.valueData()
            }
        }

        let epsilon = node.nodeDef.attr["variance_epsilon"]?.f ?? 0.001
        return BatchNorm(mean: mean, variance: variance, offset: offset, scale: scale, epsilon: epsilon, id: node.nodeDef.name)
    }

    func setupMappers() {
        // MARK: Add
        mappers[Constants.Ops.Add] = addMapper

        // MARK: Activation neurons
        mappers[Constants.Ops.Relu] = neuronMapper(.relu)
        mappers[Constants.Ops.QuantizedRelu] = neuronMapper(.relu)
        mappers[Constants.Ops.Tanh] = neuronMapper(.tanh)
        mappers[Constants.Ops.Sigmoid] = neuronMapper(.sigmoid)

        // MARK: Pooling
        mappers[Constants.Ops.MaxPool] = poolingMapper(.max)
        mappers[Constants.Ops.AvgPool] = poolingMapper(.avg)

        // MARK: SoftMax
        mappers[Constants.Ops.Softmax] = softMaxMapper

        // MARK: Conv
        mappers[Constants.Ops.Conv] = convMapper
        mappers[Constants.Ops.QuantizedConv2D] = convMapper
        if #available(iOS 11.0, *) {
            mappers[Constants.Ops.DepthwiseConv] = convMapper
        }

        // MARK: ConvTranspose
        mappers["Conv2DBackpropInput"] = convTransposeMapper

        // MARK: Dense
        mappers[Constants.Ops.Dense] = denseMapper

        // MARK: InstanceNorm
        mappers[Constants.Ops.InstanceNormAdd] = inormMapper

        // MARK: Concat
        mappers[Constants.Ops.Concat] = concatMapper
        mappers[Constants.Ops.ConcatV1] = concatMapper

        // MARK: BatchNorm
        mappers[Constants.Ops.BatchNormGlobal] = batchnormMapper
    }

}
