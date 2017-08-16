//
//  MPSImageDescriptor.swift
//  Bender
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import MetalPerformanceShadersProxy

public extension MPSImageDescriptor {
    public convenience init(layerSize: LayerSize) {
        self.init(channelFormat: .float16, width: layerSize.w, height: layerSize.h, featureChannels: layerSize.f)
    }
}

public extension MPSImage {

    var size: LayerSize {
        return LayerSize(h: height, w: width, f: featureChannels)
    }
}
