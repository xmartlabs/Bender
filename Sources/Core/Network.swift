//
//  Network.swift
//  Bender
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Represents a neural network
public class Network {

    /// Input node of the neural network.
    public var start: Start

    /// All the layers of the network
    var nodes = [NetworkLayer]()

    /// Responsible for loading the parameters
    public var parameterLoader: ParameterLoader

    /// If set to true will print information about the graph and generated dependency list
    public var verbose = false

    /// - Parameters:
    ///   - inputSize: The image size for the first layer. Input images will be resized if they do not have this size.
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    public init(inputSize: LayerSize, parameterLoader: ParameterLoader? = nil) {
        guard let device = Device.shared else {
            fatalError("Couldn't create default device")
        }
        guard MPSSupportsMTLDevice(device) else {
            fatalError("Metal Performance Shaders does not support this device \(device.description)")
        }

        start = Start(size: inputSize)
        self.parameterLoader = parameterLoader ?? NoParameterLoader()
    }

    /// Converts the graph found at `url` to its nodes
    static public func load(
        url: URL,
        converter: Converter = TFConverter.default(),
        inputSize: LayerSize,
        parameterLoader: ParameterLoader? = nil,
        performInitialize: Bool = true) -> Network {

        let network = Network(inputSize: inputSize, parameterLoader: parameterLoader)
        network.nodes = converter.convertGraph(file: url)
        if performInitialize {
            network.initialize()
        }
        return network
    }

    /// Initializes the layers of the network
    public func initialize() {
        if nodes.isEmpty {
            nodes = DependencyListBuilder().list(from: start)
        } else {
            // Add start node
            if !nodes.contains(start) {
                nodes.first?.addIncomingEdge(from: start)
                nodes.insert(start, at: 0)
            }
        }
        for layer in nodes {
            layer.initialize(network: self, device: Device.shared)
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
    ///   - dispatchQueue: the dispatch queue where to run
    ///   - callback: will be called with the output image
    public func run(
        input: MPSImage,
        queue: MTLCommandQueue? = nil,
        dispatchQueue: DispatchQueue? = nil,
        callback: @escaping (MPSImage) -> Void) {

        let commandQueue = queue ?? Device.shared.makeCommandQueue()!

        commandQueue.insertDebugCaptureBoundary() // DEBUG
        let commandBuffer = commandQueue.makeCommandBuffer()!
        commandBuffer.label = "Network run buffer"
        start.inputImage = input
        autoreleasepool {
            for layer in nodes {
                layer.execute(commandBuffer: commandBuffer)
            }
            commandBuffer.commit()
            
            if let dispatchQueue = dispatchQueue {
                dispatchQueue.async { [weak self] in
                    guard let me = self else { return }
                    
                    commandBuffer.waitUntilCompleted()
                    callback(me.nodes.last!.outputImage)
                }
            } else {
                commandBuffer.waitUntilCompleted()
                callback(nodes.last!.outputImage)
            }
        }
    }

    /// Update weights of the network.
    public func change(to checkpoint: String) {
        if checkpoint == parameterLoader.checkpoint {
            return
        }

        parameterLoader.checkpoint = checkpoint
        for layer in nodes {
            layer.updatedCheckpoint(device: Device.shared)
        }
    }

    /// Adds layers executed at the beginning of the execution list (after the Start node).
    /// Should only be used when converting graphs from other models. Is not needed if defining the network yourself.
    public func addPreProcessing(layers: [NetworkLayer]) {
        guard layers.count > 0 else { return }
        guard nodes.index(of: start) == nil else {
            fatalError("Must not call this function after initializing. Also only call after converting from a different model")
        }

        for i in 0..<layers.count-1 {
            layers[i+1].addIncomingEdge(from: layers[i])
        }

        nodes.first?.addIncomingEdge(from: layers.last!)
        nodes.insert(contentsOf: layers, at: 0)
    }

    /// Adds layers executed at the end of the execution list.
    /// Should only be used when converting graphs from other models. Is not needed if defining the network yourself.
    public func addPostProcessing(layers: [NetworkLayer]) {
        guard layers.count > 0 else { return }
        guard nodes.index(of: start) == nil else {
            fatalError("Must not call this function after initializing. Also only call after converting from a different model")
        }

        for i in 0..<layers.count-1 {
            layers[i+1].addIncomingEdge(from: layers[i])
        }

        if let output = nodes.last {
            layers.first?.addIncomingEdge(from: output)
        }
        nodes.append(contentsOf: layers)
    }

}

//extension Network: GraphProtocol {}
