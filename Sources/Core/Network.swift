//
//  Network.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders


open class Network {

    public var start: Start
    var nodes = [NetworkLayer]()
    fileprivate var device: MTLDevice
    public var parameterLoader: ParameterLoader

    public init(device: MTLDevice, inputSize: LayerSize, parameterLoader: ParameterLoader) {
        start = Start(size: inputSize)
        self.device = device
        self.parameterLoader = parameterLoader
    }

    open func initialize() {
        buildExecutionList(node: start)
        for layer in nodes {
            layer.initialize(network: self, device: device)
        }
        nodes = nodes.filter { !($0 is Dummy) }
        _ = nodes.map {
            print($0.id ?? "nil")
        }
    }

    public func run(inputImage: MPSImage, queue: MTLCommandQueue, result: @escaping (MPSImage) -> Void) {

        queue.insertDebugCaptureBoundary()
        let commandBuffer = queue.makeCommandBuffer()
        commandBuffer.label = "Network run buffer"
        start.inputImage = inputImage
        autoreleasepool {
            for layer in nodes {
                layer.execute(commandBuffer: commandBuffer)
            }
            commandBuffer.commit()
            //TODO: We should execute this on another queue
            commandBuffer.waitUntilCompleted()
            result(nodes.last!.outputImage)
        }
    }

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
