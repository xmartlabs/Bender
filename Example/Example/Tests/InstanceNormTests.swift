//
//  InstanceNormTests.swift
//  Example
//
//  Created by Mathias Claassen on 5/29/17.
//
//

import Accelerate
import AVFoundation
import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersProxy
import MetalBender

class InstanceNormTest: BenderTest {

    override func run(completion: @escaping () -> ()) {
        var tests: [CompletionSerializer.CompletableFunction] = []
        for texture in TestData.textures {
            tests.append { completion in self.test(texture: texture, completion: completion) }
//        tests.append { completion in self.test(texture: TestData.textures.first!, completion: completion) }
        }
        CompletionSerializer(completableFunctions: tests).run(completion: completion)
    }

    func test(texture: Texture, completion: @escaping () -> ()) {
        let styleNet = Network(inputSize: texture.size)
        let weights = [Float](repeating: Float.random(), count: texture.depth)
        let bias = [Float](repeating: Float.random(), count: texture.depth)
        let scale = Data(bytes: weights, count: texture.totalCount * MemoryLayout<Float>.stride)
        let shift = Data(bytes: bias, count: texture.totalCount * MemoryLayout<Float>.stride)
        styleNet.start ->> InstanceNorm(scale: scale, shift: shift)
        styleNet.initialize()
        let metalTexture = texture.metalTexture(with: Device.shared)
        let cpuComputed = cpuInstanceNorm(input: texture, weights: weights, bias: bias)
        styleNet.run(input: MPSImage(texture: metalTexture, featureChannels: texture.depth)) { image in
            let textureFromGpu = Texture(metalTexture: image.texture, size: texture.size)
            assert(textureFromGpu.isEqual(to: cpuComputed, threshold: 0.002))
            completion()
        }
    }

    public func cpuInstanceNorm(input: Texture, weights: [Float], bias: [Float]) -> Texture {
        let output = Texture(data: input.data, size: input.size)
        var mean = [Float](repeating: 0, count: input.depth)
        var variance = [Float](repeating: 0, count: input.depth)

        // Mean
        for x in 0..<input.width {
            for y in 0..<input.height {
                for z in 0..<input.depth {

                    let inputValue = input[x, y, z]
                    mean[z] += inputValue
                }
            }
        }
        for i in 0..<mean.count {
            mean[i] /= Float(input.width * input.height)
        }

        // Variance
        for x in 0..<input.width {
            for y in 0..<input.height {
                for z in 0..<input.depth {

                    let inputValue = input[x, y, z]
                    variance[z] += pow(inputValue - mean[z], 2)
                }
            }
        }
        for i in 0..<variance.count {
            let sum = variance[i] / Float(input.width * input.height)
            variance[i] = 1 / sqrt(max(sum, 1e-4) + 1e-4)

        }

        // Normalize
        for x in 0..<input.width {
            for y in 0..<input.height {
                for z in 0..<input.depth {

                    let inputValue = input[x, y, z]
                    let scale = weights[z] * variance[z]
                    let shift = bias[z] - mean[z] * scale

                    output[x, y, z] = min(max(inputValue * scale + shift, -10.0), 10.0)
                }
            }
        }
//                    var sqrSum: Float = 0
//                    for zIndex in z-parameters.depthRadius...z+parameters.depthRadius where zIndex >= 0 && zIndex < input.depth {
//                        let iterValue = input[x, y, zIndex]
//                        sqrSum += pow(iterValue, 2)
//                    }
//                    output[x, y, z] = inputValue / pow(parameters.bias + parameters.alpha * sqrSum, parameters.beta)
//                }
//            }
//        }
        return output
    }

}
