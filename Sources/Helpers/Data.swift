//
//  Data.swift
//  Bender
//
//  Created by Mathias Claassen on 5/29/17.
//
//

import Foundation

extension Data {

    func pointer<T>() -> UnsafePointer<T>? {
        return (self as NSData).bytes.assumingMemoryBound(to: T.self)
    }
    
}
