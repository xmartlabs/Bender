//
//  NetworkLayerProtocol.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import AVFoundation
import MetalPerformanceShaders

public protocol NetworkItem {

    var outputSize: LayerSize! { get }
    var outputImage: MPSImage! { get }
    func initialize(device: MTLDevice)
    func execute(commandBuffer: MTLCommandBuffer)
    func updateCheckpoint(new: String, old: String, device: MTLDevice)
    
}

public protocol Group {

    var input: NetworkLayer { get }
    var output: NetworkLayer { get }

}

typealias CompositeLayer = Group

open class NetworkLayer: NetworkItem {

    public var id: String?
    fileprivate var outgoing = [NetworkLayer]()
    fileprivate var incoming = [Weak<NetworkLayer>]()
    public var outputSize: LayerSize!
    public var outputImage: MPSImage!

    public init(id: String? = nil) {
        self.id = id
    }

    open func initialize(device: MTLDevice) {}

    open func execute(commandBuffer: MTLCommandBuffer) {
        fatalError("Not implemented")
    }

    open func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

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

//public protocol NetworkLayerUnion {
//
//    var outputSize: LayerSize! { get }
//    func initialize(device: MTLDevice, prevSizes: [LayerSize])
//    func execute(commandBuffer: MTLCommandBuffer, inputImages: [MPSImage]) -> MPSImage
//
//}

public extension NetworkLayer {

    func loadWeights(from file: String, size: Int, useFloat16: Bool = false) -> UnsafePointer<Float> {
        // Load weights from file(s)
        let typeSize = useFloat16 ? Constants.HalfSize : Constants.FloatSize
        let sizeWeights = size * typeSize

        // get the url to this layer's weights
        let wtPath = Bundle.main.path( forResource: file, ofType: "")

        // open file descriptors in read-only mode to parameter files
        let fd_w  = open( wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)

        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")

        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0);

        // cast Void pointers to Float
        let w = UnsafePointer(hdrW!.bindMemory(to: Float.self, capacity: Int(sizeWeights)))

        close(fd_w)
        assert(w != UnsafePointer<Float>.init(bitPattern: -1), "mmap failed with errno = \(errno)")
        return w
    }

    // MARK: manage incoming dependencies
    func addIncoming(layer: NetworkLayer) {
        let weakLayer = Weak(value: layer)
        if !incoming.contains(weakLayer) {
            incoming.append(weakLayer)
            if !layer.outgoing.contains(self) {
                layer.outgoing.append(self)
            }
        }
    }

    func getIncoming() -> [NetworkLayer] {
        return incoming.flatMap { $0.value }
    }

    func deleteIncoming(layer: NetworkLayer) {
        let weakLayer = Weak(value: layer)
        if let index = incoming.index(of: weakLayer) {
            incoming.remove(at: index)
        }
    }

    func getOutgoing() -> [NetworkLayer] {
        return outgoing
    }

    func deleteOutgoing(layer: NetworkLayer) {
        if let index = outgoing.index(of: layer) {
            outgoing.remove(at: index)
        }
    }

}

extension NetworkLayer: Equatable {}

public func ==(lhs: NetworkLayer, rhs: NetworkLayer) -> Bool {
    return lhs.id == rhs.id
}
