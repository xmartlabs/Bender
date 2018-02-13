//
//  Float.swift
//  Example
//
//  Created by Mathias Claassen on 2/13/18.
//

import Foundation

extension Array where Element == Float {

    static func random(count: Int) -> [Float] {
        var arr = [Float]()
        for _ in 0..<count {
            arr.append(Float(arc4random()) / Float(UINT32_MAX))
        }
        return arr
    }

    func toData(count: Int? = nil) -> Data {
        return Data(buffer: UnsafeBufferPointer(start: self, count: count ?? self.count))
    }

}
