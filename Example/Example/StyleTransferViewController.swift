//
//  ViewController.swift
//  Example
//
//  Copyright © 2017 Xmartlabs SRL. All rights reserved.
//

import Accelerate
import MetalPerformanceShadersProxy
import MetalKit
import MetalBender
import UIKit

class StyleTransferViewController: UIViewController, ExampleViewController {

    var styleNet: Network!
    var styleNet2: Network!
    var commandQueue: MTLCommandQueue!
    let inputSize = LayerSize(w: 256, f: 3)

    var pixelBufferPool: CVPixelBufferPool?
    @IBOutlet weak var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        self.commandQueue = Device.shared.makeCommandQueue()
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

            let url = Bundle.main.url(forResource: "g_and_w2", withExtension: "pb")!
            let converter = TFConverter.default(additionalOptimizers: [TFInstanceNormOptimizer()])

            styleNet = Network.load(url: url, converter: converter, inputSize: inputSize, performInitialize: false)
            styleNet.addPostProcessing(layers: [ImageLinearTransform()])

            // after adding all our layers we are able to initialize the network

            styleNet.initialize()
        }
    }

    @IBAction func runNetwork(_ sender: Any) {
        let buffer = commandQueue.makeCommandBuffer()
        let image = loadTestImage(commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        styleNet.run(input: image, queue: commandQueue) { [weak self] imageA in
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

    func loadTestImage(commandBuffer: MTLCommandBuffer) -> MPSImage{
        // INPUT IMAGE
        let textureLoader = MTKTextureLoader(device: Device.shared)
        let inputTexture = try! textureLoader.newTexture(withContentsOf: Bundle.main.url(forResource: "wall-e", withExtension: "png")!, options: [MTKTextureLoaderOptionSRGB : NSNumber(value: false)])
        return MPSImage(texture: inputTexture, featureChannels: 3)
    }

    @IBAction func stopDebug(_ sender: Any) {
        commandQueue.insertDebugCaptureBoundary()
    }

}
