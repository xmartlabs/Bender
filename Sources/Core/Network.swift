//
//  Network.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders


open class Network {

    public var start: NetworkLayer
    var nodes = [NetworkLayer]()
    fileprivate var device: MTLDevice
    private var oldCheckpoint: String

    public init(device: MTLDevice, checkpoint: String, inputSize: LayerSize) {
        start = Start(size: inputSize)
        self.device = device
        self.oldCheckpoint = checkpoint
    }

    open func initialize() {
        measure("building execution list") {
            buildExecutionList(node: start)
        }
        measure("initializing") {
            for layer in nodes {
                layer.initialize(device: device)
            }
            nodes = nodes.filter { !($0 is Dummy) }
            _ = nodes.map { print($0.id ?? "nil")}
        }
    }

//    public func run(inputImage: MPSImage, queue: MTLCommandQueue, result: @escaping (MPSImage) -> Void) {
//
//        queue.insertDebugCaptureBoundary()
//        let commandBuffer = queue.makeCommandBuffer()
//        commandBuffer.label = "Network run buffer"
//        autoreleasepool {
//            for layer in nodes {
//                layer.execute(commandBuffer: commandBuffer)
//            }
//            commandBuffer.commit()
//            //TODO: We should execute this on another queue
//            commandBuffer.waitUntilCompleted()
//            result(nodes.last!.outputImage)
//        }
//    }
//
//    public func changeToCheckpoint(checkpoint: String) {
//        if checkpoint == oldCheckpoint {
//            return
//        }
//        root.updateCheckpoint(new: checkpoint, old: oldCheckpoint, device: device)
//        oldCheckpoint = checkpoint
//    }

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
