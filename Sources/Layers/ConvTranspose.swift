//
//  ConvTransposeZerosLayer.swift
//  Bender
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2017 Xmartlabs. All rights reserved.
//

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
    var weightsPointer: Data?
    
    let size: ConvSize
    private var prevSize: LayerSize!
    
    var pipelineCalculate: MTLComputePipelineState!
    var pipelineShifLeft: MTLComputePipelineState!
    var pipelineShiftTop: MTLComputePipelineState!

    var weightsBuffer: MTLBuffer!


    /// ConvTranspoe initializer
    /// Note: padding is PaddingType.same
    ///
    /// - Parameters:
    ///   - size: Convolution size
    ///   - weights: Convolution weights
    ///   - bias: Convolution bias (not yet implemented)
    ///   - id: Node identification string. Used to load weights if they are not frozen into the graph.
    public init(size: ConvSize, weights: Data? = nil, bias: Data? = nil,
                id: String? = nil) {
        self.size = size
        self.weightsPointer = weights
        super.init(id: id)
    }
    
    open override func initialize(network: Network, device: MTLDevice) {
        super.initialize(network: network, device: device)
        let incoming = getIncoming()
        assert(incoming.count == 1, "ConvTranspose must have one input, not \(incoming.count)")
        assert(size.strideX == size.strideY, "ConvTranspose must have symmetric strides") // restriction might be taken away
        assert(size.kernelWidth == size.kernelHeight, "ConvTranspose must have symmetric kernel sizes") // restriction might be taken away
        assert(size.strideX == size.kernelWidth - 1, "ConvTranspose: stride must be kernelSize - 1")

        // Load custom metal kernels
        let constants = [FunctionConstant<ushort>(index: 0, type: MTLDataType.ushort, value: ushort(size.kernelWidth)),
                         FunctionConstant<ushort>(index: 1, type: MTLDataType.ushort, value: ushort(size.kernelHeight))]
        pipelineCalculate = MetalShaderManager.shared.getFunction(name: "transpose_conv_calculate",
                                                                  in: Bundle(for: ConvTranspose.self),
                                                                  constants: constants)
        pipelineShifLeft = MetalShaderManager.shared.getFunction(name: "transpose_conv_shift_left",
                                                                 in: Bundle(for: ConvTranspose.self),
                                                                 constants: constants)
        pipelineShiftTop = MetalShaderManager.shared.getFunction(name: "transpose_conv_shift_top",
                                                                 in: Bundle(for: ConvTranspose.self),
                                                                 constants: constants)

        prevSize = incoming[0].outputSize
        outputSize = LayerSize(f: size.outputChannels,
                               w: prevSize.w * size.strideX,
                               h: prevSize.h * size.strideY)
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: outputSize))

        weightsBuffer = device.makeBuffer(bytes: weightsPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id,
                                                                                                       modifier: ConvTranspose.weightModifier,
                                                                                                       size: getWeightsSize()),
                                          length: getWeightsSize() * Constants.FloatSize,
                                          options: [])
    }

    open override func updatedCheckpoint(device: MTLDevice) {
        guard let network = network else { return }
        let vector = weightsPointer?.pointer() ?? network.parameterLoader.loadWeights(for: id, modifier: ConvTranspose.weightModifier, size: getWeightsSize())
        weightsBuffer.contents().copyBytes(from: vector, count: getWeightsSize() * Constants.FloatSize)
    }

    open func getWeightsSize() -> Int {
        return prevSize.f * size.kernelWidth * size.kernelHeight * size.outputChannels
    }

    open override func execute(commandBuffer: MTLCommandBuffer) {

        // thread group size variables
        let incoming = getIncoming()
        let w = pipelineCalculate.threadExecutionWidth
        let d = 1
        assert(pipelineCalculate.maxTotalThreadsPerThreadgroup / w / d >= 1, "ERROR: wrong thread group size")
        let h = pipelineCalculate.maxTotalThreadsPerThreadgroup / w / d

        let step1ImageSize = LayerSize(f: outputSize.f, w: outputSize.w + prevSize.w, h: outputSize.h + prevSize.h)
        let step2ImageSize = LayerSize(f: outputSize.f, w: outputSize.w, h: outputSize.h + prevSize.h)

        let threadsPerGroups = MTLSizeMake(w, h, d)
        let threadgroupsPerGrid = MTLSize(width: (incoming[0].outputImage.texture.width + w - 1) / w,
                                          height: (incoming[0].outputImage.texture.height + h - 1) / h,
                                          depth: (outputImage.texture.arrayLength + d - 1) / d)

        let step1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step1ImageSize))

        // calculation step
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.label = "convT compute encoder"
        encoder.setComputePipelineState(pipelineCalculate)
        encoder.setTexture(incoming[0].outputImage.texture, at: 0)
        encoder.setTexture(step1Img.texture, at: 1)
        encoder.setBuffer(weightsBuffer, offset: 0, at: 0)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder.endEncoding()

        let step2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: MPSImageDescriptor(layerSize: step2ImageSize))

        // shift left step
        let encoder2 = commandBuffer.makeComputeCommandEncoder()
        encoder2.label = "convT shift left encoder"
        encoder2.setComputePipelineState(pipelineShifLeft)
        encoder2.setTexture(step1Img.texture, at: 0)
        encoder2.setTexture(step2Img.texture, at: 1)

        encoder2.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder2.endEncoding()

        step1Img.readCount = 0

        // shift top step
        let encoder3 = commandBuffer.makeComputeCommandEncoder()
        encoder3.label = "convT shift top encoder"
        encoder3.setComputePipelineState(pipelineShiftTop)
        encoder3.setTexture(step2Img.texture, at: 0)
        encoder3.setTexture(outputImage.texture, at: 1)

        encoder3.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerGroups)
        encoder3.endEncoding()

        step2Img.readCount = 0
    }

}
