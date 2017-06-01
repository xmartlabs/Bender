//
//  ViewController.swift
//  Example
//
//  Copyright © 2017 Xmartlabs SRL. All rights reserved.
//

import Accelerate
import MetalPerformanceShaders
import MetalKit
import Palladium
import UIKit

class StyleTransferViewController: UIViewController {

    var styleNet: Network!
    var styleNet2: Network!
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    let inputSize = LayerSize(f: 3, w: 256)

    private var pixelBufferPool: CVPixelBufferPool?
    @IBOutlet weak var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        self.device = MTLCreateSystemDefaultDevice()!
        self.commandQueue = device.makeCommandQueue()
        setupStyleNet()
        setPixelBufferPool()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        runNetwork(self)
    }

    private func setPixelBufferPool() {
        let bufferAttributes = [kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: Int32(kCVPixelFormatType_64RGBAHalf)),
                                kCVPixelBufferWidthKey as String: inputSize.w,
                                kCVPixelBufferHeightKey as String: inputSize.w] as [String: Any]
        CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                [kCVPixelBufferPoolMinimumBufferCountKey as String: 1] as CFDictionary,
                                bufferAttributes as CFDictionary,
                                &pixelBufferPool)
    }

    func getPixelBuffer(from texture: MTLTexture, bufferPool: CVPixelBufferPool) -> CVPixelBuffer? {
        let channels = texture.arrayLength * 4

        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, bufferPool, &pixelBuffer)

        guard let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        if let pointer = CVPixelBufferGetBaseAddress(buffer) {
            let region = MTLRegionMake2D(0, 0, inputSize.w, inputSize.h)
            texture.getBytes(pointer, bytesPerRow: 2 * channels * inputSize.w, from: region, mipmapLevel: 0)
        }
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))

        return buffer
    }

    func setupStyleNet() {
        measure("Set up takes:") {

            styleNet = Network(device: device, inputSize: inputSize, parameterLoader: nil)

            let url = Bundle.main.url(forResource: "g_and_w2", withExtension: "pb")!
            let converter = TFConverter.default()
            converter.optimizers.append(TFInstanceNormOptimizer())

            styleNet.nodes = converter.convertGraph(file: url, type: .binary)

            let imageTransform = ImageLinearTransform()
            styleNet.nodes.last! ->> imageTransform
            styleNet.nodes.append(imageTransform)

            print("\n\n\nNodes in final layer graph:")
            for node in styleNet.nodes {
                print("\(node.id)")
            }

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
