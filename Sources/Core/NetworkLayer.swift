//
//  NetworkLayerProtocol.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import AVFoundation
import MetalPerformanceShaders

/// A group is a temporary structure that represents every network component
public protocol Group {

    /// First layer of the group
    var input: NetworkLayer { get }

    /// Last layer of the group
    var output: NetworkLayer { get }

}

/// A layer composed of several sublayers. Used to reuse group of layers.
public typealias CompositeLayer = Group

open class NetworkLayer: Node {

    private static var counter = 0
    public var id: String
    public var outputSize: LayerSize!
    public var outputImage: MPSImage!
    public var network: Network?

    //MARK: Node protocol
    public var edgeOut = [Node]()
    public var edgeIn = [() -> Node?]()

    public init(id: String? = nil) {
        if let id = id {
            self.id = id
        } else {
            self.id = "Anonymous_\(NetworkLayer.counter)"
            NetworkLayer.counter += 1
        }
    }

    open func initialize(network: Network, device: MTLDevice) {
        self.network = network
    }

    open func execute(commandBuffer: MTLCommandBuffer) {
        fatalError("Not implemented")
    }

    open func updatedCheckpoint(device: MTLDevice) {}

    public func isEqual(to other: Node) -> Bool {
        return self == (other as? NetworkLayer)
    }

}

extension NetworkLayer: Group {

    public var input: NetworkLayer {
        return self
    }

    public var output: NetworkLayer {
        return self
    }

}

public struct LayerGroup: Group {

    public var input: NetworkLayer
    public var output: NetworkLayer
    
}

public extension NetworkLayer {
//
//    // MARK: manage incoming dependencies
//    func addIncoming(layer: NetworkLayer) {
//        let weakLayer = Weak(value: layer)
//        if !incoming.contains(weakLayer) {
//            incoming.append(weakLayer)
//            if !layer.outgoing.contains(self) {
//                layer.outgoing.append(self)
//            }
//        }
//    }
//
    func getIncoming() -> [NetworkLayer] {
        return incomingNodes().flatMap { $0 as? NetworkLayer }
    }
//
//    func deleteIncoming(layer: NetworkLayer) {
//        let weakLayer = Weak(value: layer)
//        if let index = incoming.index(of: weakLayer) {
//            incoming.remove(at: index)
//        }
//    }
//
    func getOutgoing() -> [NetworkLayer] {
        return outgoingNodes().flatMap { $0 as? NetworkLayer }
    }
//
//    func deleteOutgoing(layer: NetworkLayer) {
//        if let index = outgoing.index(of: layer) {
//            outgoing.remove(at: index)
//        }
//    }
//
}

extension NetworkLayer: Equatable {}

public func ==(lhs: NetworkLayer, rhs: NetworkLayer) -> Bool {
    return lhs.id == rhs.id
}
