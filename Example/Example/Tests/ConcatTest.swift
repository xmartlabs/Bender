//
//  ConcatTest.swift
//  Example
//
//  Created by Diego Ernst on 6/27/17.
//
//

import Accelerate
import AVFoundation
import MetalBender
import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersProxy

struct ConcatDataSet {

    typealias Test = (inputTextures: [Texture], axis: LayerSizeAxis, expected: Texture)

    static let testData: [Test] = [
        test1(depth: 8),
        test1(depth: 3),
        test2(depth: 9),
        test2(depth: 4),
        test3(depth1: 4, depth2: 4),
        test3(depth1: 8, depth2: 16)
    ]

    static func test1(depth: Int) -> Test {
        let texture1Depth = depth
        let texture1 = Texture(
            data: [
                [Float].init(repeating: 1, count: texture1Depth), [Float].init(repeating: 2, count: texture1Depth),
                [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 4, count: texture1Depth),
                [Float].init(repeating: 5, count: texture1Depth), [Float].init(repeating: 6, count: texture1Depth),
            ],
            size: LayerSize(h: 3, w: 2, f: texture1Depth)
        )
        let texture2Depth = depth
        let texture2 = Texture(
            data: [
                // row
                [Float].init(repeating: 1, count: texture2Depth), [Float].init(repeating: 2, count: texture2Depth),
                [Float].init(repeating: 3, count: texture2Depth),
                // row
                [Float].init(repeating: 4, count: texture2Depth), [Float].init(repeating: 5, count: texture2Depth),
                [Float].init(repeating: 6, count: texture2Depth),
                // row
                [Float].init(repeating: 7, count: texture2Depth), [Float].init(repeating: 8, count: texture2Depth),
                [Float].init(repeating: 8, count: texture2Depth),
            ],
            size: LayerSize(h: 3, w: 3, f: texture2Depth)
        )
        let expectedDepth = depth
        let expectedData: [[Float]] = [
            // row
            [Float].init(repeating: 1, count: expectedDepth), [Float].init(repeating: 2, count: expectedDepth),
            [Float].init(repeating: 1, count: expectedDepth), [Float].init(repeating: 2, count: expectedDepth),
            [Float].init(repeating: 3, count: expectedDepth),
            // row
            [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 4, count: texture1Depth),
            [Float].init(repeating: 4, count: texture2Depth), [Float].init(repeating: 5, count: texture2Depth),
            [Float].init(repeating: 6, count: texture2Depth),
            // row
            [Float].init(repeating: 5, count: texture1Depth), [Float].init(repeating: 6, count: texture1Depth),
            [Float].init(repeating: 7, count: texture2Depth), [Float].init(repeating: 8, count: texture2Depth),
            [Float].init(repeating: 8, count: texture2Depth),
        ]

        let expected: Texture = Texture(
            data: expectedData,
            size: LayerSize(h: 3, w: 5, f: expectedDepth)
        )
        return (inputTextures: [texture1, texture2], axis: .w, expected: expected)
    }

    static func test2(depth: Int) -> Test {
        let texture1Depth = depth
        let texture1 = Texture(
            data: [
                [Float].init(repeating: 1, count: texture1Depth), [Float].init(repeating: 2, count: texture1Depth),
                [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 4, count: texture1Depth),
                [Float].init(repeating: 5, count: texture1Depth), [Float].init(repeating: 6, count: texture1Depth),
            ],
            size: LayerSize(h: 3, w: 2, f: texture1Depth)
        )
        let texture2Depth = depth
        let texture2 = Texture(
            data: [
                [Float].init(repeating: 1, count: texture1Depth), [Float].init(repeating: 2, count: texture1Depth),
                [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 4, count: texture1Depth),
                [Float].init(repeating: 5, count: texture1Depth), [Float].init(repeating: 6, count: texture1Depth),
                [Float].init(repeating: 9, count: texture1Depth), [Float].init(repeating: 7, count: texture1Depth),
                [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 3, count: texture1Depth),
            ],
            size: LayerSize(h: 5, w: 2, f: texture2Depth)
        )
        let expectedDepth = depth
        let expectedData: [[Float]] = [
            [Float].init(repeating: 1, count: texture1Depth), [Float].init(repeating: 2, count: texture1Depth),
            [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 4, count: texture1Depth),
            [Float].init(repeating: 5, count: texture1Depth), [Float].init(repeating: 6, count: texture1Depth),
            [Float].init(repeating: 1, count: texture1Depth), [Float].init(repeating: 2, count: texture1Depth),
            [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 4, count: texture1Depth),
            [Float].init(repeating: 5, count: texture1Depth), [Float].init(repeating: 6, count: texture1Depth),
            [Float].init(repeating: 9, count: texture1Depth), [Float].init(repeating: 7, count: texture1Depth),
            [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 3, count: texture1Depth),
        ]

        let expected: Texture = Texture(
            data: expectedData,
            size: LayerSize(h: 8, w: 2, f: expectedDepth)
        )
        return (inputTextures: [texture1, texture2], axis: .h, expected: expected)
    }

    static func test3(depth1: Int, depth2: Int) -> Test {
        let texture1Depth = depth1
        let texture1 = Texture(
            data: [
                [Float].init(repeating: 1, count: texture1Depth), [Float].init(repeating: 2, count: texture1Depth),
                [Float].init(repeating: 3, count: texture1Depth), [Float].init(repeating: 4, count: texture1Depth),
                [Float].init(repeating: 5, count: texture1Depth), [Float].init(repeating: 6, count: texture1Depth),
            ],
            size: LayerSize(h: 3, w: 2, f: texture1Depth)
        )
        let texture2Depth = depth2
        let texture2 = Texture(
            data: [
                [Float].init(repeating: 7, count: texture2Depth), [Float].init(repeating: 8, count: texture2Depth),
                [Float].init(repeating: 9, count: texture2Depth), [Float].init(repeating: 10, count: texture2Depth),
                [Float].init(repeating: 11, count: texture2Depth), [Float].init(repeating: 12, count: texture2Depth),
            ],
            size: LayerSize(h: 3, w: 2, f: texture2Depth)
        )
        let expectedDepth = depth1 + depth2
        let expectedData: [[Float]] = [
            // row
            [Float].init(repeating: 1, count: texture1Depth) + [Float].init(repeating: 7, count: texture2Depth),
            [Float].init(repeating: 2, count: texture1Depth) + [Float].init(repeating: 8, count: texture2Depth),
            // row
            [Float].init(repeating: 3, count: texture1Depth) + [Float].init(repeating: 9, count: texture2Depth),
            [Float].init(repeating: 4, count: texture1Depth) + [Float].init(repeating: 10, count: texture2Depth),
            // row
            [Float].init(repeating: 5, count: texture1Depth) + [Float].init(repeating: 11, count: texture2Depth),
            [Float].init(repeating: 6, count: texture1Depth) + [Float].init(repeating: 12, count: texture2Depth)
        ]

        let expected: Texture = Texture(
            data: expectedData,
            size: LayerSize(h: 3, w: 2, f: expectedDepth)
        )
        return (inputTextures: [texture1, texture2], axis: .f, expected: expected)
    }

}

class ConcatTest: BenderTest {

    override func run(completion: @escaping () -> ()) {
        var tests: [CompletionSerializer.CompletableFunction] = []
        ConcatDataSet.testData.forEach { testData in
            tests.append( { completion in
                self.test(inputTextures: testData.inputTextures, axis: testData.axis, expectedOutput: testData.expected, completion: completion)
            })
        }
        CompletionSerializer(completableFunctions: tests).run(completion: completion)
    }

    func test(inputTextures: [Texture], axis: LayerSizeAxis, expectedOutput: Texture, completion: @escaping () -> ()) {
        let styleNet = Network(inputSize: inputTextures[0].size)

        styleNet.start
            ->> inputTextures.map { Constant(outputTexture: $0) }
            ->> Concat(axis: axis)

        styleNet.initialize()
        let metalTexture = inputTextures[0].metalTexture(with: Device.shared)
        styleNet.run(input: MPSImage(texture: metalTexture, featureChannels: inputTextures[0].depth)) { image in
            let textureFromGpu = Texture(metalTexture: image!.texture, size: expectedOutput.size)
            assert(textureFromGpu.isEqual(to: expectedOutput))
            completion()
        }
    }

}
