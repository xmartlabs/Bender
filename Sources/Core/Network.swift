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

    /// All the input nodes of the neural network.
    public var startNodes: [Start]

    /// First input node of the neural network.
    public var start: Start {
        return startNodes[0]
    }

    /// All the layers of the network
    public private(set) var nodes = [NetworkLayer]()

    /// Nodes for which the result will be a MPSImage instead of MPSTemporaryImage.
    /// You can use `node(for id: String)` to get hold of the nodes.
    public var permanentOutputNodes = [NetworkLayer]()

    /// Responsible for loading the parameters
    public var parameterLoader: ParameterLoader

    /// If you encode several command buffers before the previous one finishes execution then we might need to duplicate internal resources
    /// to avoid different executions to write on each others results.
    public var maxConcurrentExecutions = 1

    /// If set to true will print information about the graph and generated dependency list
    public var verbose = false

    public func node(for id: String) -> NetworkLayer? {
        return nodes.first(where: { $0.id == id })
    }

    private var initialized = false

    private var runningCount = 0

    private var descriptors: [MPSImageDescriptor]!

    /// - Parameters:
    ///   - inputSizes: An array of tuples where the first item is the identifier of an input node and the second is its size.
    /// Images will be resized if they do not have the correct size
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    public init(inputSizes: [(String, LayerSize)], parameterLoader: ParameterLoader? = nil) {
        startNodes = inputSizes.map { Start(size: $0.1, inputName: $0.0) }
        self.parameterLoader = parameterLoader ?? NoParameterLoader()
    }

    /// - Parameters:
    ///   - inputSize: The image size for the first layer. Input images will be resized if they do not have this size.
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    public init(inputSize: LayerSize, parameterLoader: ParameterLoader? = nil) {
        startNodes = [Start(size: inputSize)]
        self.parameterLoader = parameterLoader ?? NoParameterLoader()
    }

    /// Converts the graph found at `url` to its nodes
    ///
    /// - Parameters:
    ///   - url: URL to the file containing the graph
    ///   - inputSizes: array of tuples (identifier, size) for each input node of the graph
    ///   - converter: the converter to use
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    ///         Not needed if the graph includes weights
    ///   - performInitialize: if the network should be initialized. If false you will have to call network.initialize()
    /// - Returns: the converted network
    static public func load(url: URL,
                            inputSizes: [(String, LayerSize)],
                            converter: Converter = TFConverter.default(),
                            parameterLoader: ParameterLoader? = nil,
                            performInitialize: Bool = true) -> Network {

        let network = Network(inputSizes: inputSizes, parameterLoader: parameterLoader)
        network.load(url: url, converter: converter, performInitialize: performInitialize)
        return network
    }

    /// Converts the graph found at `url` to its nodes. Used if there is only one Start node
    ///
    /// - Parameters:
    ///   - url: URL to the file containing the graph
    ///   - inputSizes: size of the input node of the graph
    ///   - converter: the converter to use
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    ///         Not needed if the graph includes weights
    ///   - performInitialize: if the network should be initialized. If false you will have to call network.initialize()
    /// - Returns: the converted network
    static public func load(url: URL,
                            inputSize: LayerSize,
                            converter: Converter = TFConverter.default(),
                            parameterLoader: ParameterLoader? = nil,
                            performInitialize: Bool = true) -> Network {

        let network = Network(inputSize: inputSize, parameterLoader: parameterLoader)
        network.load(url: url, converter: converter, performInitialize: performInitialize)
        return network
    }

    func load(url: URL, converter: Converter, performInitialize: Bool) {
        set(layers: converter.convertGraph(file: url, startNodes: startNodes))

        if performInitialize {
            initialize()
        }
    }

    func set(layers: [NetworkLayer]) {
        nodes = layers
        let inputNodes = nodes.filter { $0.getIncoming().isEmpty }
        assert(inputNodes.count == startNodes.count,
               "Number of network inputs(\(inputNodes.count)) and input sizes(\(startNodes.count)) are not equal")
    }

    /// Validates that the network has been correctly converted and set up
    public func validate() {
        for layer in nodes {
            layer.validate()
        }
    }

    /// Initializes the layers of the network
    public func initialize() {
        guard let device = Device.shared else {
            fatalError("Couldn't create default device")
        }
        guard MPSSupportsMTLDevice(device) else {
            fatalError("Metal Performance Shaders does not support this device \(device.description)")
        }

        if nodes.isEmpty {
            nodes = DependencyListBuilder().list(from: startNodes as [NetworkLayer])
        }

        if let outputNode = nodes.last, !permanentOutputNodes.contains(outputNode) {
            permanentOutputNodes.append(outputNode)
        }

        // Remove dummy nodes
        nodes.forEach { ($0 as? Dummy)?.removeFromGraph() }
        nodes = nodes.filter { !($0 is Dummy) }

        // initialize layers
        for layer in nodes {
            layer.initialize(network: self, device: Device.shared, temporaryImage: !permanentOutputNodes.contains(layer))
        }

        descriptors = nodes.compactMap { $0.descriptor }

        if verbose {
            _ = nodes.map {
                debugPrint($0, " => ", $0.id)
            }
        }

        initialized = true
    }

    /// Executes the neural network
    ///
    /// - Parameters:
    ///   - inputImage: the input data
    ///   - queue: the command queue on which to run the kernels
    ///   - dispatchQueue: the dispatch queue where to run
    ///   - callback: will be called with the output image
    public func run(input: MPSImage,
                    queue: MTLCommandQueue? = nil,
                    dispatchQueue: DispatchQueue? = nil,
                    executionIndex: Int = 0,
                    callback: @escaping (MPSImage?) -> Void) {

        run(inputs: [input], queue: queue, dispatchQueue: dispatchQueue, executionIndex: executionIndex, callback: callback)
    }

    public func run(inputs: [MPSImage],
                    queue: MTLCommandQueue? = nil,
                    dispatchQueue: DispatchQueue? = nil,
                    executionIndex: Int = 0,
                    callback: @escaping (MPSImage?) -> Void) {

        guard initialized else {
            callback(nil)
            return
        }

        guard inputs.count == startNodes.count else {
            fatalError("You must pass as many inputs (" + String(inputs.count) + ") as inputSize's" + String(startNodes.count) +
                " you passed when creating the network")
        }

        // Increment running count
        runningCount += 1

        let commandQueue = queue ?? Device.shared.makeCommandQueue()!

        commandQueue.insertDebugCaptureBoundary() // DEBUG
        let commandBuffer = commandQueue.makeCommandBuffer()!
        commandBuffer.label = "Network run buffer"

        for (index, input) in inputs.enumerated() {
            startNodes[index].inputImage = input
        }

        // prefetch storage for run
        MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: descriptors)

        autoreleasepool {
            for layer in nodes {
                layer.execute(commandBuffer: commandBuffer, executionIndex: executionIndex)
            }
            commandBuffer.commit()

            if let dispatchQueue = dispatchQueue {
                dispatchQueue.async { [weak self] in
                    guard let me = self else { return }

                    commandBuffer.waitUntilCompleted()
                    callback(me.nodes.last!.getOutput(index: executionIndex))
                    me.runningCount -= 1
                }
            } else {
                commandBuffer.waitUntilCompleted()
                callback(nodes.last!.getOutput(index: executionIndex))
                runningCount -= 1
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

    /// Adds layers executed at the beginning of the execution list (after the Start node). Use only when you have a single input layer
    /// Should only be used when converting graphs from other models. Is not needed if defining the network yourself.
    public func addPreProcessing(layers: [NetworkLayer]) {
        guard !layers.isEmpty else { return }
        guard !initialized, let start = startNodes.first, nodes.firstIndex(of: start) != nil else {
            fatalError("Must not call this function after initializing. Also only call after converting from a different model")
        }

        for i in 0..<layers.count-1 {
            layers[i+1].addIncomingEdge(from: layers[i])
        }

        start.insert(outgoing: layers)

        nodes.insert(contentsOf: layers, at: 1)
    }

    /// Adds layers executed at the beginning of the execution list. The dictionary's keys are the names of the input nodes while
    /// the values are arrays of layers that will be executed before that input layer
    /// Should only be used when converting graphs from other models. Is not needed if defining the network yourself.
    public func addPreProcessing(layers: [String: [NetworkLayer]]) {
        guard !layers.isEmpty else { return }
        guard !initialized, let start = startNodes.first, nodes.firstIndex(of: start) != nil else {
            fatalError("Must not call this function after initializing. Also only call after converting from a different model")
        }
        for (inputName, preprocessing) in layers {
            for i in 0..<preprocessing.count-1 {
                preprocessing[i+1].addIncomingEdge(from: preprocessing[i])
            }

            startNode(for: inputName)?.insert(outgoing: preprocessing)
            //TODO: Study if it would be better to add these layers before their input node.
            nodes.insert(contentsOf: preprocessing, at: startNodes.count)
        }
    }

    /// Adds layers executed at the end of the execution list.
    /// Should only be used when converting graphs from other models. Is not needed if defining the network yourself.
    public func addPostProcessing(layers: [NetworkLayer]) {
        guard !layers.isEmpty else { return }
        guard !initialized else {
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

    func startNode(for name: String) -> Start? {
        return startNodes.first { $0.inputName == name }
    }

    /// Blocking function that destroys resources and uninitializes the network. Should be called before a network is to be destroyed
    /// but not from main thread
    public func destroy() {
        // Avoids new runs as side-effect
        initialized = false
        while runningCount > 0 {
            usleep(10000) // 10ms
        }
        for node in nodes {
            node.destroy()
        }
    }

}

//extension Network: GraphProtocol {}
