//
//  HelperExtensions.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 11/30/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import Foundation

public func measure(_ label: String = "", _ block: () -> ()) {
    let time1 = Date()
    block()
    let time2 = Date()
    let v = time2.timeIntervalSince(time1)
    print("\(label): \(v) (\(1/v) per second)")
}

class Weak<T: AnyObject>: Equatable where T: Equatable {
    weak var value : T?
    init (value: T) {
        self.value = value
    }
}

func ==<T: Equatable> (left: Weak<T>, right: Weak<T>) -> Bool {
    return left.value == right.value
}
