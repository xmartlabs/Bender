//
//  BatchNormTest.swift
//  Example
//
//  Created by Mathias Claassen on 2/13/18.
//

import Accelerate
import AVFoundation
import MetalBender
import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersProxy

struct BatchNormDataSet {

    static let epsilon: Float = 0.001
    static let mean = [Float].init(repeating: 0.5, count: 8)
    static let variance = [Float].init(repeating: 2, count: 8)
    static let scale = [Float].init(repeating: 1.5, count: 8)
    static let offset = [Float].init(repeating: 1, count: 8)

    static func resultWithScale(val: Float) -> Float {
        return (val - mean[0]) * scale[0] / variance[0] + offset[0]
    }

    static func resultSimple(val: Float) -> Float {
        return (val - mean[0]) / variance[0]
    }

    typealias Test = (inputTexture: Texture, useScale: Bool, expected: Texture)

    static let testData: [Test] = [
        test1(useScale: true, depth: 8),
        test1(useScale: false, depth: 8),
        test2(useScale: true, depth: 4),
        test2(useScale: false, depth: 4),
    ]

    static func test1(useScale: Bool, depth: Int = 8) -> Test {
        let texture1 = Texture(
            data: [
                [Float].init(repeating: 1, count: depth), [Float].init(repeating: 2, count: depth),
                [Float].init(repeating: 3, count: depth), [Float].init(repeating: 4, count: depth),
                [Float].init(repeating: 5, count: depth), [Float].init(repeating: 6, count: depth),
                ],
            size: LayerSize(h: 3, w: 2, f: depth)
        )

        let expected: Texture = Texture(
            data: texture1.map(op: useScale ? resultWithScale : resultSimple),
            size: LayerSize(h: 3, w: 2, f: depth)
        )
        return (inputTexture: texture1, useScale: useScale, expected: expected)
    }

    static func test2(useScale: Bool, depth: Int = 8) -> Test {
        var data = [[Float]]()
        for _ in 0..<3 {
            for _ in 0..<3 {
                data.append([Float].random(count: depth))
            }
        }
        let texture1 = Texture(data: data, size: LayerSize(h: 3, w: 3, f: depth))

        let expected: Texture = Texture(data: texture1.map(op: useScale ? resultWithScale : resultSimple),
                                        size: LayerSize(h: 3, w: 3, f: depth))
        return (inputTexture: texture1, useScale: useScale, expected: expected)
    }

}

class BatchNormTest: BenderTest {

    override func run(completion: @escaping () -> ()) {
        var tests: [CompletionSerializer.CompletableFunction] = []
        BatchNormDataSet.testData.forEach { testData in
            tests.append( { completion in
                self.test(inputTexture: testData.inputTexture,
                          useScale: testData.useScale,
                          expectedOutput: testData.expected,
                          completion: completion)
            })
        }
        CompletionSerializer(completableFunctions: tests).run(completion: completion)
    }

    func test(inputTexture: Texture, useScale: Bool, expectedOutput: Texture, completion: @escaping () -> ()) {
        let styleNet = Network(inputSize: inputTexture.size)

        styleNet.start
            ->> BatchNorm(mean: BatchNormDataSet.mean.toData(count: inputTexture.size.f),
                          variance: BatchNormDataSet.variance.toData(count: inputTexture.size.f),
                          offset: useScale ? BatchNormDataSet.offset.toData(count: inputTexture.size.f) : nil,
                          scale: useScale ? BatchNormDataSet.scale.toData(count: inputTexture.size.f) : nil,
                          epsilon: BatchNormDataSet.epsilon)

        styleNet.initialize()
        let metalTexture = inputTexture.metalTexture(with: Device.shared)
        styleNet.run(input: MPSImage(texture: metalTexture, featureChannels: inputTexture.depth)) { image in
            let textureFromGpu = Texture(metalTexture: image.texture, size: expectedOutput.size)
            if !textureFromGpu.isEqual(to: expectedOutput, threshold: 0.01) {
                print("TEST failed:")
                print(textureFromGpu)
                print(expectedOutput)
            } else {
                print("TEST PASSES")
            }
            completion()
        }
    }

}
