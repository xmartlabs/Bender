//
//  MPSImage.swift
//  Example
//
//  Created by Mathias Claassen on 2/7/18.
//

import MetalPerformanceShaders
import Accelerate

extension MPSImage {

    func toArray() -> [Float] {
        let t = Texture(metalTexture: texture, size: size)
        return t.data.flatMap { $0 }
    }

}

extension Array where Element: Comparable {
    /**
     Returns the index and value of the largest element in the array.
     */
    public func argmax() -> (Int, Element) {
        precondition(self.count > 0)
        var maxIndex = 0
        var maxValue = self[0]
        for i in 1..<self.count {
            if self[i] > maxValue {
                maxValue = self[i]
                maxIndex = i
            }
        }
        return (maxIndex, maxValue)
    }

    public func argsort(by areInIncreasingOrder: (Element, Element) -> Bool) -> [Array.Index] {
        return self.indices.sorted { areInIncreasingOrder(self[$0], self[$1]) }
    }
}
