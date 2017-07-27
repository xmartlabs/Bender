//
//  NetworkLayerProtocol.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import AVFoundation
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

    /// The result image of this layer
    public var outputImage: MPSImage!

    /// points to the network where this layer is being executed
    public weak var network: Network?

    //MARK: Node protocol
    public var edgeOut = [Node]()
    public var edgeIn = [WeakNodeClosure]()

    public init(id: String? = nil) {
        if let id = id {
            self.id = id
        } else {
            self.id = "Anonymous_\(NetworkLayer.counter)"
            NetworkLayer.counter += 1
        }
    }

    /// Initializes the layer
    open func initialize(network: Network, device: MTLDevice) {
        self.network = network
    }

    /// Runs the layer
    open func execute(commandBuffer: MTLCommandBuffer) {
        fatalError("Not implemented")
    }

    /// Used to update the weights of the layer
    open func updatedCheckpoint(device: MTLDevice) {}

    public func isEqual(to other: Node) -> Bool {
        return self == (other as? NetworkLayer)
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

extension NetworkLayer: Equatable {}

public func ==(lhs: NetworkLayer, rhs: NetworkLayer) -> Bool {
    return lhs.id == rhs.id
}
