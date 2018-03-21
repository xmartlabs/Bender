//
//  NetworkLayerProtocol.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import AVFoundation
import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// A group is a temporary structure that represents every network component
public protocol Group {

    /// First layer of the group
    var input: NetworkLayer { get }

    /// Last layer of the group
    var output: NetworkLayer { get }

}

/// A layer composed of several sublayers. Used to reuse group of layers.
public typealias CompositeLayer = Group

/// A layer (node) of the neural network
open class NetworkLayer: Node {

    private static var counter = 0

    /// Identifies the network layer
    public var id: String

    /// Size of the layers outputImage
    public var outputSize: LayerSize!

    /// The result images of this layer if permanent
    private var outputs: [MPSImage]

    /// The output image if temporary
    private var outputImage: MPSTemporaryImage?

    /// Descriptor for temporary images
    var descriptor: MPSImageDescriptor?

    /// points to the network where this layer is being executed
    public weak var network: Network?

    // MARK: Node protocol
    public var edgeOut = [Node]()
    public var edgeIn = [WeakNodeClosure]()

    public init(id: String? = nil) {
        if let id = id {
            self.id = id
        } else {
            self.id = "Anonymous_\(NetworkLayer.counter)"
            NetworkLayer.counter += 1
        }
        outputs = []
    }

    /// Validates the correctness of a layers inputs and parameters
    open func validate() {}

    /// Initializes the layer
    open func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        self.network = network
        validate()
    }

    /// Creates output images or structures for the layer
    public func createOutputs(size: LayerSize, temporary: Bool = true) {
        if temporary {
            descriptor = MPSImageDescriptor(layerSize: size)
            descriptor?.storageMode = .private
        } else {
            guard let maxConcurrentExecutions = network?.maxConcurrentExecutions,
                    maxConcurrentExecutions > 0 else {
                        assertionFailure("Layer has no network while being initialized")
                        return
            }
            for _ in 0..<maxConcurrentExecutions {
                outputs.append(MPSImage(device: Device.shared, imageDescriptor: MPSImageDescriptor(layerSize: size)))
            }
        }
    }

    /// Gets the output image for a layer
    public func getOutput(index: Int = 0) -> MPSImage {
        if let outputImage = outputImage {
            return outputImage
        } else {
            assert(index < outputs.count)
            return outputs[index]
        }
    }

    /// Gets the output image for the layer if it exists. Creates it otherwise.
    public func getOrCreateOutput(commandBuffer: MTLCommandBuffer, index: Int = 0) -> MPSImage {
        if let descriptor = descriptor {
            if outputImage == nil || outputImage?.readCount == 0 {
                outputImage = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: descriptor)
                outputImage?.readCount = getOutgoing().count
            }

            return outputImage!
        } else {
            assert(index < outputs.count)
            return outputs[index]
        }
    }

    /// Sets the layers output (at specified index if applicable) to be the specified image.
    /// This is often used in layers that don't change the input like the Identity.
    public func rewireIdentity(at index: Int, image: MPSImage) {
        if let temp = image as? MPSTemporaryImage {
            outputImage = temp
            outputImage?.readCount = getOutgoing().count
        } else {
            if outputs.isEmpty {
                createOutputs(size: outputSize, temporary: false)
            }
            outputs[index] = image
        }
    }

    /// Runs the layer
    ///
    /// - Parameters:
    ///   - commandBuffer: MTLCommandBuffer that will run the layer
    ///   - executionIndex: Execution index for concurrent executions. Index for output images
    open func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        fatalError("Not implemented")
    }

    /// Used to update the weights of the layer
    open func updatedCheckpoint(device: MTLDevice) {}

    public func isEqual(to other: Node) -> Bool {
        return self == (other as? NetworkLayer)
    }

    func destroy() {
        if let outputImage = outputImage, outputImage.readCount != 0 {
            outputImage.readCount = 0
        }
        outputs.removeAll()
    }

}

extension NetworkLayer: Group {

    /// To conform to Group, NetworkLayer is a group of [self] so `input` is self
    public var input: NetworkLayer {
        return self
    }

    /// To conform to Group, NetworkLayer is a group of [self] so `output` is self
    public var output: NetworkLayer {
        return self
    }

}

/// Concrete implementation of Group that serves to create groups of layers when building the network. Mainly used by the operators
public struct LayerGroup: Group {

    public var input: NetworkLayer
    public var output: NetworkLayer

}

public extension NetworkLayer {

    /// Helper method that casts the incoming edges to NetworkLayer
    func getIncoming() -> [NetworkLayer] {
        return incomingNodes().flatMap { $0 as? NetworkLayer }
    }

    /// Helper method that casts the outgoing edges to NetworkLayer
    func getOutgoing() -> [NetworkLayer] {
        return outgoingNodes().flatMap { $0 as? NetworkLayer }
    }

}

extension NetworkLayer: Equatable {
    public static func == (lhs: NetworkLayer, rhs: NetworkLayer) -> Bool {
        return lhs.id == rhs.id
    }
}
