//
//  Pooling.swift
//  Bender
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Type for Pooling layer
public enum PoolingType {

    case max
    case avg

}

/// Implements MaxPooling and AvgPooling.
open class Pooling: NetworkLayer {

    public var pooling: MPSCNNPooling!
    public var padding: PaddingType
    var type: PoolingType
    var kernelSize: (width: Int, height: Int)
    var stride: (x: Int, y: Int)

    public init(type: PoolingType, padding: PaddingType = .same, kernelSize: (width: Int, height: Int) = (2, 2),
                stride: (x: Int, y: Int) = (2, 2), id: String? = nil) {
        self.type = type
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        super.init(id: id)
    }

    open override func validate() {
        // Check correctness
        let incoming = getIncoming()
        assert(incoming.count == 1, "Pooling must have one input, not \(incoming.count)")
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        let prevSize = incoming[0].outputSize!

        // Set up pooling
        switch type {
        case .max:
            self.pooling = MPSCNNPoolingMax(device: device,
                                            kernelWidth: kernelSize.width,
                                            kernelHeight: kernelSize.height,
                                            strideInPixelsX: stride.x,
                                            strideInPixelsY: stride.y)
        case.avg:
            self.pooling = MPSCNNPoolingAverage(device: device,
                                                kernelWidth: kernelSize.width,
                                                kernelHeight: kernelSize.height,
                                                strideInPixelsX: stride.x,
                                                strideInPixelsY: stride.y)
        }
        self.pooling.edgeMode = .clamp

        // Set up padding
        switch padding {
        case .same:
            self.pooling.offset.x += (((prevSize.w - 1) % pooling.strideInPixelsX) / 2) + (pooling.kernelWidth + 1) % 2
            self.pooling.offset.y += (((prevSize.h - 1) % pooling.strideInPixelsY) / 2) + (pooling.kernelHeight + 1) % 2
            outputSize = LayerSize(h: prevSize.h / pooling.strideInPixelsY,
                                   w: prevSize.w / pooling.strideInPixelsX,
                                   f: prevSize.f)
        case .valid:
            self.pooling.offset.x += Int(pooling.kernelWidth)/2
            self.pooling.offset.y += Int(pooling.kernelHeight)/2
            outputSize = LayerSize(h: (prevSize.h - pooling.kernelHeight) / pooling.strideInPixelsY + 1,
                                   w: (prevSize.w - pooling.kernelWidth) / pooling.strideInPixelsX + 1,
                                   f: prevSize.f)
        }

        createOutputs(size: outputSize)
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex: Int = 0) {
        pooling.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputs[executionIndex], destinationImage: outputs[executionIndex])
    }

}
