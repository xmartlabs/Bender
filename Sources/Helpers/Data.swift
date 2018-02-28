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

    func toFloat16() -> Data? {
        if let pointer: UnsafePointer<Float> = self.pointer() {
            let mutable = UnsafeMutablePointer(mutating: pointer)
            return float32to16(mutable, count: count / Constants.FloatSize).toData()
        }
        return nil
    }

    func toArray<T>() -> [T] {
        return withUnsafeBytes { (pointer: UnsafePointer<T>) -> [T] in
            let buffer = UnsafeBufferPointer(start: pointer,
                                             count: self.count / MemoryLayout<T>.size)
            return [T](buffer)
        }
    }

}
