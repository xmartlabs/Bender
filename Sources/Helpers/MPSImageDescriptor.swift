//
//  MPSImageDescriptor.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/3/17.
//
//

import MetalPerformanceShaders

extension MPSImageDescriptor {
    convenience init(layerSize: LayerSize) {
        self.init(channelFormat: .float16, width: layerSize.w, height: layerSize.h, featureChannels: layerSize.f)
    }
}
