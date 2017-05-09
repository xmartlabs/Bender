//
//  MTLDevice.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/9/17.
//
//

import Metal

extension MTLDevice {

    func makeMyLibrary() -> MTLLibrary {
        guard let path = Bundle(for: NetworkLayer.self).path(forResource: "default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }

        return try! makeLibrary(filepath: path)
    }
    
}
