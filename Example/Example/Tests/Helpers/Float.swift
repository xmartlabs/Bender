//
//  Float.swift
//  Example
//
//  Created by Mathias Claassen on 2/13/18.
//

import Foundation

extension Array where Element == Float {

    static func random(count: Int) -> [Float] {
        return (0..<count).map { _ in Float(arc4random()) / Float(UINT32_MAX) }
    }

}
