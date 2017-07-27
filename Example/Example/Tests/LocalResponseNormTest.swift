//
//  LocalResponseNormTest.swift
//  Bender
//
//  Created by Diego Ernst on 5/25/17.
//
//

import Accelerate
import AVFoundation
import MetalKit
import MetalPerformanceShadersProxy
import Bender

class LocalResponseNormTest: BenderTest {

    let testParameters: [LocalResponseNorm.Parameters] = [
        LocalResponseNorm.Parameters(),
        LocalResponseNorm.Parameters(depthRadius: 5, bias: 2, alpha: 3, beta: 4),
        LocalResponseNorm.Parameters(depthRadius: 10, bias: 5, alpha: 6, beta: 7),
        LocalResponseNorm.Parameters(depthRadius: 20, bias: 8, alpha: 9, beta: 10)
    ]

    override func run(completion: @escaping (Void) -> ()) {
        var tests: [CompletionSerializer.CompletableFunction] = []
        for params in testParameters {
            for texture in TestData.textures {
                tests.append { completion in self.test(texture: texture, parameters: params, completion: completion) }
            }
        }
        CompletionSerializer(completableFunctions: tests).run(completion: completion)
    }

    func test(texture: Texture, parameters: LocalResponseNorm.Parameters, completion: @escaping (Void) -> ()) {
        let styleNet = Network(device: device, inputSize: texture.size, parameterLoader: SingleBinaryLoader(checkpoint: "lala"))
        styleNet.start ->> LocalResponseNorm(parameters: parameters, id: nil)
        styleNet.initialize()
        let metalTexture = texture.metalTexture(with: device)
        let cpuComputed = cpuLocalResponseNorm(input: texture, parameters: parameters)
        styleNet.run(inputImage: MPSImage(texture: metalTexture, featureChannels: texture.depth), queue: device.makeCommandQueue()) { image in
            let textureFromGpu = Texture(metalTexture: image.texture, size: texture.size)
            assert(textureFromGpu.isEqual(to: cpuComputed, threshold: 0.001))
            completion()
        }
    }

    public func cpuLocalResponseNorm(input: Texture, parameters: LocalResponseNorm.Parameters) -> Texture {
        let output = Texture(data: input.data, size: input.size)
        for x in 0..<input.width {
            for y in 0..<input.height {
                for z in 0..<input.depth {

                    let inputValue = input[x, y, z]
                    var sqrSum: Float = 0
                    for zIndex in z-parameters.depthRadius...z+parameters.depthRadius where zIndex >= 0 && zIndex < input.depth {
                        let iterValue = input[x, y, zIndex]
                        sqrSum += pow(iterValue, 2)
                    }
                    output[x, y, z] = inputValue / pow(parameters.bias + parameters.alpha * sqrSum, parameters.beta)
                }
            }
        }
        return output
    }

}
