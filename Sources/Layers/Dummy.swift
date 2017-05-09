//
//  Dummy.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/9/17.
//
//

import MetalPerformanceShaders

open class Dummy: NetworkLayer {

    open override func initialize(device: MTLDevice) {
        let incoming = getIncoming()
        let outgoing = getOutgoing()
        for out in outgoing {
            out.deleteIncoming(layer: self)
        }

        for inc in incoming {
            inc.deleteOutgoing(layer: self)
            for out in outgoing {
                out.addIncoming(layer: inc)
            }
        }
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {
        fatalError("Dummy in execution list")
    }
    
}
