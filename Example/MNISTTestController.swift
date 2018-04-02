//
//  MNISTTestController.swift
//  Example
//
//  Created by Mathias Claassen on 5/30/17.
//
//

import AVFoundation
import CoreGraphics
import MetalBender
import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersProxy
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

    var commandQueue: MTLCommandQueue!

    //MARK: ExampleViewController
    let inputSize = LayerSize(h: 28, w: 28, f: 3)
    var pixelBufferPool: CVPixelBufferPool?

    var captureSession = AVCaptureSession()
    var previewLayer = AVCaptureVideoPreviewLayer()
    var movieOutput: AVCaptureVideoDataOutput!
    lazy var videoDevice: AVCaptureDevice! = {
        let x = AVCaptureDevice.DiscoverySession(deviceTypes: [AVCaptureDevice.DeviceType.builtInDuoCamera, .builtInWideAngleCamera, .builtInTelephotoCamera], mediaType: AVMediaType.video, position: AVCaptureDevice.Position.back).devices
        return x.first
    }()

    fileprivate var textureCache: CVMetalTextureCache?

    override func viewDidLoad() {
        super.viewDidLoad()
        self.commandQueue = Device.shared.makeCommandQueue()
        setupTextureCache()
        setupCaptureSession()
        setupScaledNetwork()

        // Import network
        importMNISTNetwork()

        // If you want to create it from scratch comment the line above and uncomment:
        // createMNISTNetwork()
        
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
        mtkView.device = Device.shared
        mtkView.delegate = self
        mtkView.clearColor = MTLClearColorMake(0, 0, 0, 0)
        mtkView.framebufferOnly = false
        mtkView.autoResizeDrawable = true
        mtkView.drawableSize = CGSize(width: inputSize.w, height: inputSize.h)

        pipeline = MetalShaderManager.shared.getFunction(name: "to_grayscale")

    }

    func setupCaptureSession() {
        let capturePreset = AVCaptureSession.Preset.cif352x288
        guard videoDevice.supportsSessionPreset(capturePreset) else {
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
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        previewLayer.connection?.videoOrientation = AVCaptureVideoOrientation.portrait
        cameraView.layer.addSublayer(previewLayer)

        captureSession.commitConfiguration()

        try? videoDevice.lockForConfiguration()

        let duration = Int64(100)
        videoDevice.activeVideoMinFrameDuration = CMTime(value: duration, timescale: 1000)

        videoDevice.unlockForConfiguration()
    }

    func setupTextureCache() {
        guard CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, Device.shared, nil, &textureCache) == kCVReturnSuccess else {
            debugPrint("Error: Could not create a texture cache")
            return
        }
    }

    //MARK: Set up network
    func importMNISTNetwork() {
        // Import the MNIST network from a TF exported graph
        let url = Bundle.main.url(forResource: "mnist_full", withExtension: "pb")!
        let converter = TFConverter.default(verbose: true)

        network = Network.load(url: url, inputSize: inputSize, converter: converter, performInitialize: false)
        network.addPreProcessing(layers: [GrayScale(), Neuron(type: .custom(neuron: MPSCNNNeuronLinear(device: Device.shared, a: 255, b: 0)))])
        network.addPostProcessing(layers: [Softmax()])
        network.verbose = true

        network.initialize()
    }

    func createMNISTNetwork() {
        // Create the MNIST network with a weight file
        Convolution.weightModifier = "_w"
        Convolution.biasModifier = "_b"
        FullyConnected.weightModifier = "_w"
        FullyConnected.biasModifier = "_b"

        let parameterLoader = PerLayerBinaryLoader(checkpoint: "mnist-")
        network = Network(inputSize: inputSize, parameterLoader: parameterLoader)

        network.start ->> GrayScale()
            ->> Neuron(type: .custom(neuron: MPSCNNNeuronLinear(device: Device.shared, a: 255, b: 0)))
            ->> Convolution(convSize: ConvSize(outputChannels: 32, kernelSize: 5, stride: 1), neuronType: .relu, useBias: true, id: "conv1")
            ->> Pooling(type: .max)
            ->> Convolution(convSize: ConvSize(outputChannels: 64, kernelSize: 5, stride: 1), neuronType: .relu, useBias: true, id: "conv2")
            ->> Pooling(type: .max)
            ->> FullyConnected(neurons: 1024, neuronType: .relu, useBias: true, id: "fc1")
            ->> FullyConnected(neurons: 10, neuronType: .relu, useBias: true, id: "fc2")
            ->> Softmax()

        network.initialize()
    }

    func setupScaledNetwork() {
        // This network is used to display the texture as it is passed to the main MNIST network
        scaledNetwork = Network(inputSize: inputSize)
        scaledNetwork.initialize()
    }

    func runNetwork(_ image: MPSImage) {
        networkRunQueue.async { [weak self] in
            self?.network.run(input: image, queue: self!.commandQueue) { [weak self] results in
                let numbers = Texture(metalTexture: results!.texture, size: LayerSize(h: 1, w: 1, f: 10))
                self?.didScan(numbers: numbers.data.flatMap { $0 })
            }

            self?.scaledNetwork.run(input: image, queue: self!.commandQueue) { [weak self] (outputImage) in
                self?._texture = outputImage!.texture
            }
        }
    }

    func didScan(numbers: [Float]) {
        guard numbers.count == 10 else { return }
        DispatchQueue.main.async { [weak self] in
            if let max = numbers.max(), let index = numbers.index(of: max) {
                self?.resultLabel.text = "Found \(index)"
            } else {
                self?.resultLabel.text = "None"
            }
        }

    }
    
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension MNISTTestController: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard 1 == CMSampleBufferGetNumSamples(sampleBuffer) else {
            return
        }

        guard connection.videoOrientation == .portrait else {
            connection.videoOrientation = .portrait
            return
        }

        if let texture = createTexture(from: sampleBuffer) {
            runNetwork(MPSImage(texture: texture, featureChannels: 3))
        }
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
        let buffer = commandQueue.makeCommandBuffer()!
        let drawableTexture = (view.currentDrawable as? CAMetalDrawable)?.texture
        let encoder = buffer.makeComputeCommandEncoder()!
        encoder.label = "GrayScale encoder"
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(_texture, index: 0)
        encoder.setTexture(drawableTexture, index: 1)
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
