//
//  PalladiumTest.swift
//  Palladium
//
//  Created by Diego Ernst on 5/25/17.
//
//

import MetalKit

public class PalladiumTest {

    var device: MTLDevice!

    init() {
        device = MTLCreateSystemDefaultDevice()
    }

    func run(completion: @escaping (Void) -> ()) {
        completion()
    }

}

public class PalladiumTestRunner {

    let tests: [PalladiumTest] = [
        TextureConversionTest(),
        LocalResponseNormTest()
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
