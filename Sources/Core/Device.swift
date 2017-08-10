//
//  Device.swift
//  Bender
//
//  Created by Diego Ernst on 8/14/17.
//
//

import Metal

public struct Device {

    /// Represents the singleton instance of the GPU
    public static let shared: MTLDevice! = {
        return MTLCreateSystemDefaultDevice()
    }()

}
