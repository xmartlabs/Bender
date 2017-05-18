//
//  Padding.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/11/17.
//
//

import MetalPerformanceShaders

public enum PaddingType {

    /// Tensorflow's SAME padding. Pads evenly with left-overs added on right and bottom.
    case same // pads

    /// Tensorflow's VALID padding. No padding
    case valid // no padding

}
