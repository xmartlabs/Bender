//
//  Network.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders


class Network {

    var root: Pipeline
    fileprivate var device: MTLDevice
    private var oldCheckpoint: String

    init(device: MTLDevice, root: Pipeline, checkpoint: String) {
        self.device = device
        self.root = root
        self.oldCheckpoint = checkpoint
    }

    func initialize(inputSize: LayerSize) {
        root.initialize(device: device, prevSize: inputSize)
    }

    public func run(inputImage: MPSImage, queue: MTLCommandQueue, result: @escaping (MPSImage) -> Void) {

        queue.insertDebugCaptureBoundary()
        let commandBuffer = queue.makeCommandBuffer()
        commandBuffer.label = "Network run buffer"
        autoreleasepool {
            let image = root.execute(commandBuffer: commandBuffer, inputImage: inputImage)
            commandBuffer.commit()
            //TODO: We should execute this on another queue
            commandBuffer.waitUntilCompleted()
            result(image)
        }
    }

    public func changeToCheckpoint(checkpoint: String) {
        if checkpoint == oldCheckpoint {
            return
        }
        root.updateCheckpoint(new: checkpoint, old: oldCheckpoint, device: device)
        oldCheckpoint = checkpoint
    }

}
