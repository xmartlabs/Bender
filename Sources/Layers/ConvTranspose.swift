//
//  ConvTransposeZerosLayer.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import MetalPerformanceShadersProxy

/// Transpose 2D Convolution (conv2d_transpose in TF). Not Deconvolution.
/// Three-step implementation: See more info about this in conv_transpose.metal
/// Current Limitations:
/// * Symmetric strides and kernel sizes.
/// * Stride must be: stride == kernelSize - 1.
/// * More than 4 feature channels for input and output. (This ought to be the easiest to support)
open class ConvTranspose: NetworkLayer {

    /// Used to determine the filename for this layers weights. (Ignored if there is no ParameterLoader)
    static var weightModifier: String = ""
    static var biasModifier: String = ""
    var weightsPointer: Data?
    var biasPointer: Data?
    var useBias: Bool

    let size: ConvSize
    private var prevSize: LayerSize!

    // Own implementation
    var pipelineCalculate: MTLComputePipelineState!
    var pipelineShiftLeft: MTLComputePipelineState!
    var pipelineShiftTop: MTLComputePipelineState!
    var weightsBuffer: MTLBuffer!

    // MPSCNN implementation
    var cnnDescriptor: MPSCNNConvolutionDescriptor!
    var dataSource: Any?
    var conv: Any?

    /// ConvTranspoe initializer
    /// Note: padding is PaddingType.same
    ///
    /// - Parameters:
    ///   - size: Convolution size
    ///   - weights: Convolution weights
    ///   - bias: Convolution bias (only iOS 11.3.1+)
    ///   - id: Node identification string. Used to load weights if they are not frozen into the graph.
    public init(size: ConvSize, weights: Data? = nil, bias: Data? = nil, useBias: Bool = false, id: String? = nil) {
        self.size = size
        self.weightsPointer = weights
        self.biasPointer = bias
        self.useBias = useBias
        super.init(id: id)
    }

    open override func validate() {
        let incoming = getIncoming()
        assert(incoming.count == 1, "ConvTranspose must have one input, not \(incoming.count)")
        if #available(iOS 11.3.1, *) {} else {
            assert(size.strideX == size.strideY, "ConvTranspose must have symmetric strides") // restriction might be taken away
            assert(size.kernelWidth == size.kernelHeight, "ConvTranspose must have symmetric kernel sizes") // restriction might be taken away
            assert(size.strideX == size.kernelWidth - 1, "ConvTranspose: stride must be kernelSize - 1")
        }
    }

    open override func initialize(network: Network, device: MTLDevice, temporaryImage: Bool = true) {
        super.initialize(network: network, device: device, temporaryImage: temporaryImage)
        let incoming = getIncoming()
        prevSize = incoming[0].outputSize
        outputSize = LayerSize(h: prevSize.h * size.strideY,
                               w: prevSize.w * size.strideX,
                               f: size.outputChannels)

        createOutputs(size: outputSize, temporary: temporaryImage)

        if #available(iOS 11.3.1, *) {
            createCNNDescriptor(device: device)
            makeConv(device: device)
        } else {
            // Load custom metal kernels
            let constants = [FunctionConstant<ushort>(index: 0, type: MTLDataType.ushort, value: ushort(size.kernelWidth)),
                             FunctionConstant<ushort>(index: 1, type: MTLDataType.ushort, value: ushort(size.kernelHeight))]
            pipelineCalculate = MetalShaderManager.shared.getFunction(name: "transpose_conv_calculate",
                                                                      in: Bundle(for: ConvTranspose.self),
                                                                      constants: constants)
            pipelineShiftLeft = MetalShaderManager.shared.getFunction(name: "transpose_conv_shift_left",
                                                                     in: Bundle(for: ConvTranspose.self),
                                                                     constants: constants)
            pipelineShiftTop = MetalShaderManager.shared.getFunction(name: "transpose_conv_shift_top",
                                                                     in: Bundle(for: ConvTranspose.self),
                                                                     constants: constants)
            weightsBuffer = device.makeBuffer(bytes: weightsPointer?.pointer() ??
                                        network.parameterLoader.loadWeights(for: id,
                                                                            modifier: ConvTranspose.weightModifier,
                                                                            size: getWeightsSize()),
                                              length: getWeightsSize() * Constants.FloatSize,
                                              options: [])
        }
    }

    @available(iOS 11.3.1, *)
    open func createCNNDescriptor(device: MTLDevice) {
        //TODO: Add neuron here?
        cnnDescriptor = MPSCNNConvolutionDescriptor(kernelWidth: size.kernelWidth,
                                                    kernelHeight: size.kernelHeight,
                                                    inputFeatureChannels: prevSize.f,
                                                    outputFeatureChannels: size.outputChannels,
                                                    neuronFilter: nil)
        cnnDescriptor.strideInPixelsX = size.strideX
        cnnDescriptor.strideInPixelsY = size.strideY
        cnnDescriptor.dilationRateX = size.dilationX
        cnnDescriptor.dilationRateY = size.dilationY
    }

    @available(iOS 11.0, *)
    open func makeConv(device: MTLDevice) {
        guard let network = network else { return }
        let dataSource = ConvolutionDataSource(cnnDescriptor: cnnDescriptor,
                                               weights: UnsafeMutableRawPointer(mutating: weightsPointer?.pointer() ??
                                                network.parameterLoader.loadWeights(for: id,
                                                                                    modifier: ConvTranspose.weightModifier,
                                                                                    size: getWeightsSize())),
                                               bias: useBias ? UnsafeMutablePointer(mutating: biasPointer?.pointer() as UnsafePointer<Float>? ??
                                                network.parameterLoader.loadWeights(for: id,
                                                                                    modifier: ConvTranspose.biasModifier,
                                                                                    size: size.outputChannels)) : nil)
        conv = MPSCNNConvolutionTranspose(device: device, weights: dataSource)
        self.dataSource = dataSource
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        guard let network = network else { return }
        let vector = weightsPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id,
                                                                                      modifier: ConvTranspose.weightModifier,
                                                                                      size: getWeightsSize())
        if #available(iOS 11.3.1, *) {
            makeConv(device: device)
        } else {
            weightsBuffer.contents().copyMemory(from: vector, byteCount: getWeightsSize() * Constants.FloatSize)
        }
    }

    open func getWeightsSize() -> Int {
        return prevSize.f * size.kernelWidth * size.kernelHeight * size.outputChannels
    }

    open override func execute(commandBuffer: MTLCommandBuffer, executionIndex index: Int = 0) {

        // thread group size variables
        let incoming = getIncoming()
        let input = incoming[0].getOutput(index: index)
        let output = getOrCreateOutput(commandBuffer: commandBuffer, index: index)

        if #available(iOS 11.3.1, *), let conv = conv as? MPSCNNConvolutionTranspose {
            conv.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: output)
        } else {
            let w = pipelineCalculate.threadExecutionWidth
            let d = 1
            assert(pipelineCalculate.maxTotalThreadsPerThreadgroup / w / d >= 1, "ERROR: wrong thread group size")
            let h = pipelineCalculate.maxTotalThreadsPerThreadgroup / w / d

            let step1ImageSize = LayerSize(h: outputSize.h + prevSize.h, w: outputSize.w + prevSize.w, f: outputSize.f)
            let step2ImageSize = LayerSize(h: outputSize.h + prevSize.h, w: outputSize.w, f: outputSize.f)

            let threadsPerGroups = MTLSizeMake(w, h, d)
            let threadgroupsPerGrid = input.texture.threadGrid(threadGroup: threadsPerGroups)

            let step1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step1ImageSize))

            // calculation step
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.label = "convT compute encoder"
            encoder.setComputePipelineState(pipelineCalculate)
            encoder.setTexture(input.texture, index: 0)
            encoder.setTexture(step1Img.texture, index: 1)
            encoder.setBuffer(weightsBuffer, offset: 0, index: 0)

            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
            encoder.endEncoding()

            input.setRead()

            let step2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step2ImageSize))

            // shift left step
            let encoder2 = commandBuffer.makeComputeCommandEncoder()!
            encoder2.label = "convT shift left encoder"
            encoder2.setComputePipelineState(pipelineShiftLeft)
            encoder2.setTexture(step1Img.texture, index: 0)
            encoder2.setTexture(step2Img.texture, index: 1)

            encoder2.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
            encoder2.endEncoding()

            step1Img.readCount = 0

            // shift top step
            let encoder3 = commandBuffer.makeComputeCommandEncoder()!
            encoder3.label = "convT shift top encoder"
            encoder3.setComputePipelineState(pipelineShiftTop)
            encoder3.setTexture(step2Img.texture, index: 0)
            encoder3.setTexture(output.texture, index: 1)

            encoder3.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
            encoder3.endEncoding()

            step2Img.readCount = 0
        }
    }

}
