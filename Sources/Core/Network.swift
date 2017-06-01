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

    /// Input node of the neural network.
    public var start: Start

    /// All the layers of the network
    public var nodes = [NetworkLayer]()

    /// Responsible for loading the parameters
    public var parameterLoader: ParameterLoader

    /// If set to true will print information about the graph and generated dependency list
    public var verbose = false

    fileprivate var device: MTLDevice

    /// - Parameters:
    ///   - device: the MTLDevice.
    ///   - inputSize: The image size for the first layer. Input images will be resized if they do not have this size.
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    public init(device: MTLDevice, inputSize: LayerSize, parameterLoader: ParameterLoader?) {
        start = Start(size: inputSize)
        self.device = device
        self.parameterLoader = parameterLoader ?? NoParameterLoader()
    }

    /// Initializes the layers of the network
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


    /// Executes the neural network
    ///
    /// - Parameters:
    ///   - inputImage: the input data
    ///   - queue: the command queue on which to run the kernels
    ///   - callback: will be called with the output image
    public func run(inputImage: MPSImage, queue: MTLCommandQueue, callback: @escaping (MPSImage) -> Void) {

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
            callback(nodes.last!.outputImage)
        }
    }


    /// Update weights of the network.
    public func change(to checkpoint: String) {
        if checkpoint == parameterLoader.checkpoint {
            return
        }

        parameterLoader.checkpoint = checkpoint
        for layer in nodes {
            layer.updatedCheckpoint(device: device)
        }
    }

    /// Takes the graph and builds a dependency list (list of nodes in the order in which they will be executed)
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
