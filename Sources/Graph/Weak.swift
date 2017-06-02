//
//  Weak.swift
//  Bender
//
//  Created by Mathias Claassen on 5/17/17.
//
//

import Foundation

protocol WeakClass: Any, class {}

class Weak<T: WeakClass>: Equatable where T: Equatable {
    weak var value : T?
    init (value: T) {
        self.value = value
    }
}

//class Weak<T: WeakClass>: _Weak<T>, Equatable where T: Equatable {}


func ==<T: Equatable> (left: Weak<T>, right: Weak<T>) -> Bool {
    return left.value == right.value
}
