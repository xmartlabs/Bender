//
//  ConvolutionLayer.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders

enum ActivationNeuronType {
    case relu
    case tanh
    case none
}

public class ConvolutionLayer: NetworkLayer {
    
    let size: ConvSize
    private var prevSize: LayerSize!
    var outputLayerSize: LayerSize {
        get { return size.layerSize }
    }
    
    var descriptor: MPSImageDescriptor?
    var padding: Bool
    var conv: SlimMPSCNNConvolution?
    
    var weightsFiles: [String]?
    var kernelWeights: UnsafePointer<Float>?
    var inormScaleWeights: MTLBuffer!
    var inormShiftWeights: MTLBuffer!
    
    let neuronType: ActivationNeuronType
    var neuron: MPSCNNNeuron?
    
    let instanceNorm: Bool
    
    var outputImages = [MPSImage]()
 
    convenience init(size: ConvSize, neuronType: ActivationNeuronType = .relu, weightsFiles: String...) {
        self.init(size: size, neuronType: neuronType, weightsFiles: weightsFiles)
    }
    
    init(size: ConvSize, neuronType: ActivationNeuronType = .relu, weightsFiles: [String], padding: Bool = true) {
        self.size = size
        self.neuronType = neuronType
        self.weightsFiles = weightsFiles
        self.descriptor = MPSImageDescriptor(layerSize: size.layerSize)
        self.padding = padding
        self.instanceNorm = true
    }
    
    func initialize(device: MTLDevice, prevSize: LayerSize) {
        self.prevSize = prevSize
        if let weightsFiles = weightsFiles {
            kernelWeights = loadConvWeights(fromFilePath: weightsFiles[0], prevSize: prevSize, size: size)
            inormScaleWeights = device.makeBuffer(
                bytes: loadVectorWeights(fromFilePath: weightsFiles[1], channels: size.layerSize.f),
                length: max(4, size.layerSize.f) * Constants.FloatSize,
                options: [])
            inormShiftWeights = device.makeBuffer(
                bytes: loadVectorWeights(fromFilePath: weightsFiles[2], channels: size.layerSize.f),
                length: max(4, size.layerSize.f) * Constants.FloatSize,
                options: [])
        }
        
        makeConv(device: device)
        
        switch neuronType {
        case .relu:
            neuron = MPSCNNNeuronReLU(device: device, a: 0)
        case .tanh:
            neuron = MPSCNNNeuronTanH(device: device, a: 1, b: 1)
        default:
            neuron = nil
        }
        
        // We need up to three output images
        // - Conv output
        // - Instance norm output
        // - Neuron output (if used)
        outputImages.append(createImage(device: device))
        outputImages.append(createImage(device: device))
        if neuron != nil {
            outputImages.append(createImage(device: device))
        }
    }

    func updateCheckpoint(new checkpoint: String, old: String, device: MTLDevice) {

        for index in 0..<weightsFiles!.count {
            weightsFiles?[index] = weightsFiles![index].replacingOccurrences(of: old, with: checkpoint, options: String.CompareOptions.anchored)
        }

        if var weightsFiles = weightsFiles {
            kernelWeights = loadConvWeights(fromFilePath: weightsFiles[0], prevSize: prevSize, size: size)
            inormScaleWeights.contents().copyBytes(from: loadVectorWeights(fromFilePath: weightsFiles[1], channels: size.layerSize.f), count: max(4, size.layerSize.f) * Constants.FloatSize)
            inormShiftWeights.contents().copyBytes(from: loadVectorWeights(fromFilePath: weightsFiles[2], channels: size.layerSize.f), count: max(4, size.layerSize.f) * Constants.FloatSize)

            makeConv(device: device)
        }
    }

    func makeConv(device: MTLDevice) {
        let desc = MPSCNNConvolutionDescriptor(
            kernelWidth: size.kernelSize,
            kernelHeight: size.kernelSize,
            inputFeatureChannels: prevSize.f,
            outputFeatureChannels: size.layerSize.f,
            neuronFilter: nil)

        desc.strideInPixelsX = size.stride
        desc.strideInPixelsY = size.stride

        conv = SlimMPSCNNConvolution(device: device,
                                     convolutionDescriptor: desc,
                                     kernelWeights: kernelWeights!,
                                     biasTerms: nil,
                                     flags: .none)
    }
    
    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage, originalImage: MPSImage?) -> MPSImage {
        conv?.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImages[0], padding: self.padding)
        
        if (instanceNorm) {
            ConvolutionLayer.instanceNorm(commandBuffer: commandBuffer,
                                          inputImage: self.outputImages[0],
                                          size: self.size.layerSize,
                                          outputImage: self.outputImages[1],
                                          scaleBuffer: inormScaleWeights,
                                          shiftBuffer: inormShiftWeights)
            
            if let neuron = neuron {
                neuron.encode(commandBuffer: commandBuffer, sourceImage: outputImages[1], destinationImage: outputImages[2])
                return outputImages[2]
            } else {
                return outputImages[1]
            }
        } else {
            if let neuron = neuron {
                neuron.encode(commandBuffer: commandBuffer, sourceImage: outputImages[0], destinationImage: outputImages[2])
                return outputImages[2]
            } else {
                return outputImages[0]
            }
        }
        
    }
    
    func createImage(device: MTLDevice) -> MPSImage {
        return MPSImage(device: device, imageDescriptor: descriptor!)
    }
    
}
