//
//  MNISTTestController.swift
//  Example
//
//  Created by Mathias Claassen on 5/30/17.
//
//

import AVFoundation
import CoreGraphics
import MetalPerformanceShaders
import MetalKit
import Palladium
import UIKit

class MNISTTestController: UIViewController, ExampleViewController {

    @IBOutlet weak var cameraView: UIView!
    @IBOutlet weak var resultLabel: UILabel!

    @IBOutlet weak var mtkView: MTKView!
    var pipeline: MTLComputePipelineState!
    var _texture: MTLTexture!
    
    var network: Network!
    var scaledNetwork: Network!
    private var networkRunQueue: DispatchQueue!

    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!

    //MARK: ExampleViewController
    let inputSize = LayerSize(f: 3, w: 28)
    var pixelBufferPool: CVPixelBufferPool?

    var captureSession = AVCaptureSession()
    var previewLayer = AVCaptureVideoPreviewLayer()
    var movieOutput: AVCaptureVideoDataOutput!
    lazy var videoDevice: AVCaptureDevice! = {
        let x = AVCaptureDeviceDiscoverySession(deviceTypes: [AVCaptureDeviceType.builtInDuoCamera, .builtInWideAngleCamera, .builtInTelephotoCamera], mediaType: AVMediaTypeVideo, position: AVCaptureDevicePosition.back).devices!
        return x.first
    }()

    fileprivate var textureCache: CVMetalTextureCache?
    var run = false

    override func viewDidLoad() {
        super.viewDidLoad()
        self.device = MTLCreateSystemDefaultDevice()!
        self.commandQueue = device.makeCommandQueue()
        setupTextureCache()
        setupCaptureSession()
        setupNetwork()
        var me = self
        me.setPixelBufferPool()
        setupMetalView()
        networkRunQueue = DispatchQueue.global(qos: .background)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        captureSession.startRunning()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = cameraView.bounds
    }

    func setupMetalView() {
        mtkView.device = device
        mtkView.delegate = self
        mtkView.clearColor = MTLClearColorMake(0, 0, 0, 0)
        mtkView.framebufferOnly = false
        mtkView.autoResizeDrawable = true
        mtkView.drawableSize = CGSize(width: inputSize.w, height: inputSize.h)

        pipeline = MetalShaderManager.shared.getFunction(name: "to_grayscale")

    }

    func setupCaptureSession() {
        let capturePreset = AVCaptureSessionPreset352x288
        guard videoDevice.supportsAVCaptureSessionPreset(capturePreset) else {
            print("Device does not support medium quality")
            return
        }

        let outputSettings: [String: Any] = [String(describing: kCVPixelBufferPixelFormatTypeKey): NSNumber(value: kCVPixelFormatType_32BGRA)]

        captureSession.beginConfiguration()

        // Set video quality
        captureSession.sessionPreset = capturePreset

        //add device inputs (front camera and mic)
        try? captureSession.addInput(AVCaptureDeviceInput(device: videoDevice))

        //add output to get the frames
        movieOutput = AVCaptureVideoDataOutput()
        movieOutput.videoSettings = outputSettings
        movieOutput.setSampleBufferDelegate(self, queue: DispatchQueue.global(qos: .background))
        movieOutput.alwaysDiscardsLateVideoFrames = true
        captureSession.addOutput(movieOutput)

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = AVLayerVideoGravityResizeAspectFill
        previewLayer.connection.videoOrientation = AVCaptureVideoOrientation.portrait
        cameraView.layer.addSublayer(previewLayer)

        captureSession.commitConfiguration()

        try? videoDevice.lockForConfiguration()

        let duration = Int64(100)
        videoDevice.activeVideoMinFrameDuration = CMTime(value: duration, timescale: 1000)

        videoDevice.unlockForConfiguration()
    }

    func setupTextureCache() {
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache) == kCVReturnSuccess else {
            debugPrint("Error: Could not create a texture cache")
            return
        }
    }

    func setupNetwork() {
        measure("Set up takes:") {

            network = Network(device: device, inputSize: inputSize, parameterLoader: nil)
            network.verbose = true
            let url = Bundle.main.url(forResource: "mnist", withExtension: "pb")!
            let converter = TFConverter.default()
            converter.verbose = true

            network.convert(converter: converter, url: url, type: .binary)
            network.addPreProcessing(layers: [GrayScale()])
            network.addPostProcessing(layers: [Softmax()])

            network.initialize()

            scaledNetwork = Network(device: device, inputSize: inputSize, parameterLoader: nil)
            scaledNetwork.initialize()
        }
    }

    func runNetwork(_ image: MPSImage) {
//        let time = Date()
        networkRunQueue.async { [weak self] in
            self?.network.run(inputImage: image, queue: self!.commandQueue) { [weak self] results in
//                let v = Date().timeIntervalSince(time)
    //            debugPrint("Run: \(v) (\(1/v) per second)")
                let numbers = Texture(metalTexture: results.texture, size: LayerSize(f: 10, w: 1, h: 1))
                self?.didScan(numbers: numbers.data.flatMap { $0 })
            }

            self?.scaledNetwork.run(inputImage: image, queue: self!.commandQueue) { [weak self] (outputImage) in
                self?._texture = outputImage.texture
            }
        }
    }

    func didScan(numbers: [Float]) {
        guard numbers.count == 10 else { return }
//        print("Scanned \(numbers)")
        DispatchQueue.main.async { [weak self] in
            if let max = numbers.max(), let index = numbers.index(of: max) {
                self?.resultLabel.text = "Found \(index)"
            } else {
                self?.resultLabel.text = "None"
            }
        }

    }

    @IBAction func stopDebug(_ sender: Any) {
        commandQueue.insertDebugCaptureBoundary()
    }

    @IBAction func runDebug(_ sender: Any) {
        run = true
    }
    
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension MNISTTestController: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {

        guard 1 == CMSampleBufferGetNumSamples(sampleBuffer) else {
            return
        }

        run = false

        guard connection.videoOrientation == .portrait else {
            connection.videoOrientation = .portrait
            return
        }

        if let texture = createTexture(from: sampleBuffer) {
            runNetwork(MPSImage(texture: texture, featureChannels: 3))
        }

    }

    func captureOutput(_ captureOutput: AVCaptureOutput!, didDrop sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {

    }

    func createTexture(from sampleBuffer: CMSampleBuffer) -> MTLTexture? {
        if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer), let textureCache = textureCache {
            let width = CVPixelBufferGetWidth(imageBuffer)
            let height = CVPixelBufferGetHeight(imageBuffer)

            var cvtexture: CVMetalTexture?
            CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, textureCache,
                                                      imageBuffer, nil, .bgra8Unorm, width, height, 0, &cvtexture)

            return CVMetalTextureGetTexture(cvtexture!)
        } else {
            debugPrint("samplebuffer is nil \(sampleBuffer)")
        }
        
        return nil
    }

}

extension MNISTTestController: MTKViewDelegate {

    func draw(in view: MTKView) {
        if _texture == nil {
            return
        }
        let buffer = commandQueue.makeCommandBuffer()
        let drawableTexture = view.currentDrawable?.texture
        let encoder = buffer.makeComputeCommandEncoder()
        encoder.label = "GrayScale encoder"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(_texture, at: 0)
        encoder.setTexture(drawableTexture, at: 1)
        let threadsPerGroups = MTLSizeMake(32, 4, 1)
        let threadGroups = _texture.threadGrid(threadGroup: threadsPerGroups)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        buffer.present(view.currentDrawable!)
        buffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }

}
