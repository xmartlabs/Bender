//
//  BenderTest.swift
//  Bender
//
//  Created by Diego Ernst on 5/25/17.
//
//

import MetalKit

public class BenderTest {

    func run(completion: @escaping () -> ()) {
        completion()
    }

}

public class BenderTestRunner {

    let tests: [BenderTest] = [
        TextureConversionTest(),
        LocalResponseNormTest(),
        InstanceNormTest(),
        ConcatTest()
    ]

    public init() {}

    public func run() {
        CompletionSerializer(completableFunctions: tests.map { $0.run }) {
            print("Running test: \(String(describing: self.tests[$0]))")
        }
        .run {
            print("ALL TEST PASSED")
        }
    }

}
