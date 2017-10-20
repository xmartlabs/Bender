//
//  CompletionSerializer.swift
//  Example
//
//  Created by Diego Ernst on 5/26/17.
//
//

import Foundation

class CompletionSerializer {

    typealias CompletableFunction = (@escaping () -> ()) -> ()
    private let willStartHandler: ((Int) -> Void)?
    private let completableFunctions: [CompletableFunction]

    init(completableFunctions: [CompletableFunction], willStartHandler: ((Int) -> Void)? = nil) {
        self.completableFunctions = completableFunctions
        self.willStartHandler = willStartHandler
    }

    func run(completion: @escaping () -> ()) {
        recursiveRun(index: 0, completion: completion)
    }

    private func recursiveRun(index: Int, completion: @escaping () -> ()) {
        if index < completableFunctions.count {
            willStartHandler?(index)
            completableFunctions[index]() { [weak self] in
                self?.recursiveRun(index: index + 1, completion: completion)
            }
        } else {
            completion()
        }
    }

}
