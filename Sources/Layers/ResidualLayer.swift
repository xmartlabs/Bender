//
//  ResidualBlockLayer.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShadersProxy

/** Generic Residual layer. This is a composite layer.
 */
open class ResidualLayer: CompositeLayer {

    public var input: NetworkLayer
    public var output: NetworkLayer
    
    public init(convSize: ConvSize, layers: Group, useBias: Bool = false, id: String? = nil) {
        // We need a Dummy here because the Residual starts with a fork. When the network is initialized both of the layers on the fork will be connected to the residual's input. The dummy helps with pointer management.
        let resId = id ?? ""
        let residual = Dummy() // gets removed after graph is complete
                       ->> [Identity(id: resId + "ID"),
                            layers]
                        ->> Add(id: resId + "ADD")

        self.input = residual.input
        self.output = residual.output
    }
    
}

