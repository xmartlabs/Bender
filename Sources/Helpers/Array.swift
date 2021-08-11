//
//  Array.swift
//  MetalBender
//
//  Created by Mathias Claassen on 2/14/18.
//

import Foundation

public extension Array {

    func toData(count: Int? = nil) -> Data {
        return withUnsafePointer(to: self) {
            return Data(buffer: UnsafeBufferPointer(start: $0, count: Swift.min(count ?? self.count, self.count)))
        }
    }

    func cleanMap<ElementOfResult>(_ block: (Element) -> ElementOfResult?) -> [ElementOfResult] {
        #if swift(>=4.1)
            return compactMap(block)
        #else
            return flatMap(block)
        #endif
    }

}
