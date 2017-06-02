//
//  MTLDevice.swift
//  Bender
//
//  Created by Mathias Claassen on 5/9/17.
//
//

import Metal

extension MTLDevice {

    func makeMyLibrary(bundle: Bundle) -> MTLLibrary {
        guard let path = bundle.path(forResource: "default", ofType: "metallib") else {
            fatalError("Metal library not found")
        }

        return try! makeLibrary(filepath: path)
    }
    
}
