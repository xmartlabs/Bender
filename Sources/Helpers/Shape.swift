//
//  Shape.swift
//  Bender
//
//  Created by Mathias Claassen on 5/31/17.
//
//
public struct Shape {

    let width: Int
    let height: Int
    let inputChannels: Int
    let outputChannels: Int

    var totalCount: Int {
        return [width, height, inputChannels, outputChannels].reduce(1) { $0 * $1 }
    }

}

extension Shape {

    func at(_ index: Int) -> Int {
        switch index {
        case 0:
            return self.height
        case 1:
            return self.width
        case 2:
            return self.inputChannels
        case 3:
            return self.outputChannels
        default:
            fatalError()
        }
    }

}
