//
//  TFConverter.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/18/17.
//
//

import Foundation

public typealias TFMapper = (TensorflowNode) -> NetworkLayer

public class TFConverter: Converter {

    public var optimizers: [TFOptimizer]
    public var mappers = [String: TFMapper]()

    public init(optimizers: [TFOptimizer]) {
        self.optimizers = optimizers
    }

    public static func `default`() -> TFConverter {
        let instance = TFConverter(optimizers: [TFVariableProcessor()])
        instance.setupMappers()
        return instance
    }

    public func convertGraph(file: URL, type: ProtoFileType) -> [NetworkLayer] {
        let loader = TensorflowGraphLoader()
        var graph = loader.load(file: file, type: type)
        print("\n\n\nNodes in my graph (pre optimization):")
        for node in graph.nodes {
            print("\(node.nodeDef.name)")
        }

        runOptimizers(graph: &graph)
        return translateOperators(graph: graph)
    }

    func runOptimizers(graph: inout TensorflowGraph) {
        for optimizer in optimizers {
            optimizer.optimize(graph: graph)
            graph.removeLonely()
        }

        print("\n\n\nNodes in my graph (after optimization):")
        for node in graph.nodes {
            print("\(node.nodeDef.name)")
        }
    }

    func translateOperators(graph: TensorflowGraph) -> [NetworkLayer] {
        var layers = [NetworkLayer]()
        var processed = [String: NetworkLayer]() // maps TF node names to layers (to link nodes)

        for node in graph.nodes {
            if let mapper = mappers[node.nodeDef.op] {
                let layer = mapper(node)
                layers.append(layer)
                processed[node.nodeDef.name] = layer
                // I get the indexes for all the input nodes and link the corresponding network layers
                // For this to work we must add the Dummies when we cannot translate an `op`
                for input in node.incomingNodes() {
                    // Following unwrap will never fail because all nodes are TensorflowNode
                    // If clause is false only when the input node could not be processed because we do not support it.
                    // This is because graphs are dependency ordered and therefore dependencies will always be processed before.
                    if let inputLayer = processed[(input as! TensorflowNode).nodeDef.name] {
                        layer.addIncomingEdge(from: inputLayer)
                    }
                }
            } else {
                // We found an unsupported layer. We ignore it but warn and add a dummy to maintain the indexes of the nodes
                debugPrint("Palladium:: Unsupported layer found: \(node.nodeDef.op)")
            }
        }

        return layers
    }
}

extension TFConverter {

    func setupMappers() {
        //MARK: Add
        let addMapper = { (node: TensorflowNode) in
            return Add(id: node.nodeDef.name)
        }
        mappers["Add"] = addMapper

        //MARK: Relu
        let reluMapper = { (node: TensorflowNode) in
            return Neuron(type: ActivationNeuronType.relu, id: node.nodeDef.name)
        }
        mappers["Relu"] = reluMapper

        //MARK: Tanh
        let tanhMapper = { (node: TensorflowNode) in
            return Neuron(type: ActivationNeuronType.tanh, id: node.nodeDef.name)
        }
        mappers["Tanh"] = tanhMapper

        //MARK: Conv
        let convMapper = { (node: TensorflowNode) -> NetworkLayer in
            guard let pad = node.nodeDef.attr["padding"]?.s,
                let padString = String(data: pad, encoding: .utf8),
                let strides = node.nodeDef.attr["strides"]?.list.i,
                let dataFormat = node.nodeDef.attr["data_format"]?.s,
                let formatString = String(data: dataFormat, encoding: .utf8) else {
                fatalError("Cannot create Conv2D")
            }

            let strideX = formatString == "NHWC" ? strides[2] : strides[3]
            let strideY = formatString == "NHWC" ? strides[1] : strides[2]

            guard let weightData = TFWeightData.getWeightData(node: node) else {
                fatalError("Could not get weight information for this Conv2DTranspose")
            }

            //transpose weights
            let weights = weightData.weights// != nil ? HWIOtoOHWI(weights: weightData.weights!, shape: weightData.weightShape) : nil

            let convSize = ConvSize(shape: weightData.weightShape,
                                    strideX: Int(strideX),
                                    strideY: Int(strideY))
            return Convolution(convSize: convSize,
                               neuronType: .none,
                               useBias: weightData.bias != nil,
                               padding: PaddingType.fromTF(padString),
                               weights: weights,
                               bias: weightData.bias,
                               id: node.nodeDef.name)
        }
        mappers["Conv2D"] = convMapper

        //MARK: ConvTranspose
        let convTransposeMapper = { (node: TensorflowNode) -> NetworkLayer in
            guard let strides = node.nodeDef.attr["strides"]?.list.i,
                let dataFormat = node.nodeDef.attr["data_format"]?.s,
                let formatString = String(data: dataFormat, encoding: .utf8) else {
                    fatalError("Cannot create Conv2DTranspose")
            }

            let strideX = formatString == "NHWC" ? strides[2] : strides[3]
            let strideY = formatString == "NHWC" ? strides[1] : strides[2]

            guard let weightData = TFWeightData.getWeightData(node: node) else {
                fatalError("Could not get weight information for this Conv2DTranspose")
            }

            let convSize = ConvSize(shape: weightData.weightShape,
                                    strideX: Int(strideX),
                                    strideY: Int(strideY))
            return ConvTranspose(size: convSize,
                          weights: weightData.weights,
                          bias: weightData.bias,
                          id: node.nodeDef.name)
        }
        mappers["Conv2DBackpropInput"] = convTransposeMapper

    }

}
