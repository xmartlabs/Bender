//
//  Pooling.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShaders

public enum PoolingType {

    case max
    case avg

}

open class Pooling: NetworkLayer {

    public var pooling: MPSCNNPooling
    public var padding: PaddingType

    public init(device: MTLDevice, type: PoolingType, padding: PaddingType = .same, kernelSize: (width: Int, height: Int) = (2, 2), stride: (x: Int, y: Int) = (2, 2), id: String? = nil) {
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
        self.padding = padding
        super.init(id: id)
    }

    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "Pooling must have one input, not \(incoming.count)")
        let prevSize = incoming[0].outputSize!
        switch padding {
        case .same:
            self.pooling.offset.x += (((prevSize.w - 1) % pooling.strideInPixelsX) / 2) + (pooling.kernelWidth + 1) % 2
            self.pooling.offset.y += (((prevSize.h - 1) % pooling.strideInPixelsY) / 2) + (pooling.kernelHeight + 1) % 2
            outputSize = LayerSize(f: prevSize.f,
                                   w: prevSize.w / pooling.strideInPixelsX,
                                   h: prevSize.h / pooling.strideInPixelsY)
        case .valid:
            self.pooling.offset.x += Int(pooling.kernelWidth)/2
            self.pooling.offset.y += Int(pooling.kernelHeight)/2
            outputSize = LayerSize(f: prevSize.f,
                                   w: (prevSize.w - pooling.kernelWidth) / pooling.strideInPixelsX + 1,
                                   h: (prevSize.w - pooling.kernelHeight) / pooling.strideInPixelsY + 1)
        }

        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))
    }


    open override func execute(commandBuffer: MTLCommandBuffer) {
        pooling.encode(commandBuffer: commandBuffer, sourceImage: getIncoming()[0].outputImage, destinationImage: outputImage)
    }
    
}
