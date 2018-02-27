//
//  Array.swift
//  MetalBender
//
//  Created by Mathias Claassen on 2/14/18.
//

import Foundation

public extension Array {

    func toData(count: Int? = nil) -> Data {
        return Data(buffer: UnsafeBufferPointer(start: self, count: Swift.min(count ?? self.count, self.count)))
    }

}
