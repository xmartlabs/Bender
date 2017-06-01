//
//  Network.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders


/// Represents a neural network
public class Network: GraphProtocol {

    public var start: Start
    public var nodes = [NetworkLayer]()
    fileprivate var device: MTLDevice
    public var parameterLoader: ParameterLoader
    public var verbose = false

    ///
    /// - Parameters:
    ///   - device: the MTLDevice.
    ///   - inputSize: The image size for the first layer. Input images will be resized if they do not have this size.
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    public init(device: MTLDevice, inputSize: LayerSize, parameterLoader: ParameterLoader?) {
        start = Start(size: inputSize)
        self.device = device
        self.parameterLoader = parameterLoader ?? NoParameterLoader()
    }

    public func initialize() {
        if nodes.isEmpty {
            buildExecutionList(node: start)
        } else {
            // Add start node
            if !nodes.contains(start) {
                nodes.first?.addIncomingEdge(from: start)
                nodes.insert(start, at: 0)
            }
        }
        for layer in nodes {
            layer.initialize(network: self, device: device)
        }
        nodes = nodes.filter { !($0 is Dummy) }

        if verbose {
            _ = nodes.map {
                debugPrint($0.id)
            }
        }
    }

    public func run(inputImage: MPSImage, queue: MTLCommandQueue, result: @escaping (MPSImage) -> Void) {

        queue.insertDebugCaptureBoundary() // DEBUG
        let commandBuffer = queue.makeCommandBuffer()
        commandBuffer.label = "Network run buffer"
        start.inputImage = inputImage
        autoreleasepool {
            for layer in nodes {
                layer.execute(commandBuffer: commandBuffer)
            }
            commandBuffer.commit()
            //TODO: We should execute this on another dispatch queue
            commandBuffer.waitUntilCompleted()
            result(nodes.last!.outputImage)
        }
    }


    /// Update weights of the network.
    ///
    public func change(to checkpoint: String) {
        if checkpoint == parameterLoader.checkpoint {
            return
        }

        parameterLoader.checkpoint = checkpoint
        for layer in nodes {
            layer.updatedCheckpoint(device: device)
        }
    }

    func buildExecutionList(node: NetworkLayer) {
        guard !node.getIncoming().contains (where: { incoming in
            return !nodes.contains(incoming)
        }) else { return }
        nodes.append(node)
        for node in node.getOutgoing() {
            buildExecutionList(node: node)
        }
    }

}
