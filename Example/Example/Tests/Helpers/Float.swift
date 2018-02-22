//
//  Float.swift
//  Example
//
//  Created by Mathias Claassen on 2/13/18.
//

import Foundation

public protocol Randomizable {
    static func random() -> Self
}

extension Float : Randomizable {
    public static func random() -> Float {
        return Float(arc4random()) / Float(UInt32.max)
    }
}

extension Array where Element : Randomizable {

    static func random(count: Int) -> [Element] {
        return (0..<count).map { _ in Element.random() }
    }

}
