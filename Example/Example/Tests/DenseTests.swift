//
//  DenseTests.swift
//  Example
//
//  Created by Mathias Claassen on 3/8/18.
//

import Accelerate
import AVFoundation
import MetalBender
import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersProxy

struct DenseDataSet {

    static let input: [Float] = [0.49597502, 0.9758464,  0.49763316, 0.43710703, 0.4187885, 0.0204866, 0.74651086, 0.02312839, 0.02699029,
                                 0.38462982, 0.28852502, 0.571153, 0.27879143, 0.30243126, 0.74364275, 0.30527815, 0.8603831,  0.8035045,
                                 0.31146497, 0.47445327, 0.9097413, 0.9146943,  0.7386387,  0.85666627]

    static let result: Texture = Texture(data: [[14.026826, 14.469796, 15.204336, 15.124344, 14.45108, 14.704981, 14.454319, 14.52294,  15.053391, 14.929831]], size: LayerSize(h: 1, w: 1, f: 10))

}

@available(iOS 11.0, *)
class DenseTest: BenderTest {

    var network: Network!

    override func run(completion: @escaping () -> ()) {
        let url = Bundle.main.url(forResource: "denseTest", withExtension: "pb")!
        let converter = TFConverter.default()

        let sizes = [("dense1/BiasAdd", LayerSize(h: 4, w: 2, f: 3))]

        let inputImage = MPSImage(device: Device.shared, imageDescriptor: MPSImageDescriptor(channelFormat: .float16, width: 2, height: 4, featureChannels: 3))
        inputImage.writeBytes(float32to16(UnsafeMutablePointer(mutating: DenseDataSet.input), count: 24),
                              dataLayout: .HeightxWidthxFeatureChannels, imageIndex: 0)

        network = Network.load(url: url, inputSizes: sizes, converter: converter)
        network.run(inputs: [inputImage]) { result in
            let out = Texture(metalTexture: result.texture, size: LayerSize(h: 1, w: 1, f: 10))
            assert(out.isEqual(to: DenseDataSet.result, threshold: 0.01))
            completion()
        }
    }

}
