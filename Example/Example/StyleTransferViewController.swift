//
//  ViewController.swift
//  Example
//
//  Copyright © 2017 Xmartlabs SRL. All rights reserved.
//

import Accelerate
import MetalPerformanceShadersProxy
import MetalKit
import Bender
import UIKit

class StyleTransferViewController: UIViewController, ExampleViewController {

    var styleNet: Network!
    var styleNet2: Network!
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    let inputSize = LayerSize(f: 3, w: 256)

    var pixelBufferPool: CVPixelBufferPool?
    @IBOutlet weak var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        self.device = MTLCreateSystemDefaultDevice()!
        self.commandQueue = device.makeCommandQueue()
        setupStyleNet()
        var me = self
        me.setPixelBufferPool()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        runNetwork(self)
    }

    func setupStyleNet() {
        measure("Set up takes:") {

            styleNet = Network(device: device, inputSize: inputSize, parameterLoader: nil)

            let url = Bundle.main.url(forResource: "g_and_w2", withExtension: "pb")!
            let converter = TFConverter.default()
            converter.optimizers.append(TFInstanceNormOptimizer())

            styleNet.convert(converter: converter, url: url, type: .binary)
            styleNet.addPostProcessing(layers: [ImageLinearTransform()])

            styleNet.initialize()
        }
    }

    @IBAction func runNetwork(_ sender: Any) {
        let buffer = commandQueue.makeCommandBuffer()
        let image = loadTestImage(device: device, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        styleNet.run(inputImage: image, queue: commandQueue) { [weak self] imageA in
            if let buffer = self?.getPixelBuffer(from: imageA.texture, bufferPool: self!.pixelBufferPool!) {
                let ciImage = CIImage(cvImageBuffer: buffer)
                let context = CIContext()
                let cgImage = context.createCGImage(ciImage, from: ciImage.extent)
                let uiImage = UIImage(cgImage: cgImage!)
                DispatchQueue.main.async {
                    self?.imageView.image = uiImage
                }
            }
        }
    }

    func loadTestImage(device: MTLDevice, commandBuffer: MTLCommandBuffer) -> MPSImage{
        // INPUT IMAGE
        let textureLoader = MTKTextureLoader(device: device)
        let inputTexture = try! textureLoader.newTexture(withContentsOf: Bundle.main.url(forResource: "wall-e", withExtension: "png")!, options: [MTKTextureLoaderOptionSRGB : NSNumber(value: false)])
        return MPSImage(texture: inputTexture, featureChannels: 3)
    }

    @IBAction func stopDebug(_ sender: Any) {
        commandQueue.insertDebugCaptureBoundary()
    }

}
