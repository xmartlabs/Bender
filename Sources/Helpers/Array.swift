//
//  Array.swift
//  MetalBender
//
//  Created by Mathias Claassen on 2/14/18.
//

import Foundation

public extension Array {

    func toData(count: Int? = nil) -> Data {
        return withUnsafeBytes { pointer in
            let elements = Swift.min(count ?? self.count, self.count)
            return Data(bytes: pointer.baseAddress!,
                        count: Int(elements * MemoryLayout<Self.Element>.size))
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
