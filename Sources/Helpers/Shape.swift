//
//  Shape.swift
//  Bender
//
//  Created by Mathias Claassen on 5/31/17.
//
//

import Foundation

public struct Shape {

    let width: Int
    let height: Int
    let inputChannels: Int
    let outputChannels: Int

    var totalCount: Int {
        return [width, height, inputChannels, outputChannels].reduce(1) { $0 * $1 }
    }

}
