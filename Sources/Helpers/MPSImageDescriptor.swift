//
//  MPSImageDescriptor.swift
//  Bender
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

public extension MPSImageDescriptor {
    convenience init(layerSize: LayerSize) {
        self.init(channelFormat: .float16, width: layerSize.w, height: layerSize.h, featureChannels: layerSize.f)
    }
}

public extension MPSImage {

    var size: LayerSize {
        return LayerSize(h: height, w: width, f: featureChannels)
    }

    func setRead() {
        if let `self` = self as? MPSTemporaryImage, self.readCount != 0 {
            self.readCount -= 1
        }
    }

}
