//
//  StyleNet.swift
//  VGGNet
//
//  Created by Joaquin Rocco on 11/22/16.
//  Copyright © 2016 Hollance. All rights reserved.
//

import AVFoundation
import MetalPerformanceShaders

public class StyleNet {
    
    let device: MTLDevice
    
    var network: NetworkPipeline
    private var oldCheckpoint: String
    
    public init(device: MTLDevice, checkpoint: String) {
        self.device = device
        self.oldCheckpoint = checkpoint
        
        network = NetworkPipeline(device: device, inputSize: LayerSize(f:0, w:0), layers: [
            CropAndFormatLayer(device: device),
            ConvolutionLayer(size: ConvSize(layerSize: LayerSize(f: Constants.baseKernelCount, w: Constants.inputImageWidth), kernelSize: 9, stride: 1),
                             weightsFiles:  checkpoint+"-conv1",
                             checkpoint+"-conv1-instanceNormScale",
                             checkpoint+"-conv1-instanceNormShift"),
            ConvolutionLayer(size: ConvSize(layerSize: LayerSize(f: Constants.baseKernelCount * 2, w: Constants.inputImageWidth/2),
                                            kernelSize: 3, stride: 2),
                             weightsFiles:  checkpoint+"-conv2",
                                            checkpoint+"-conv2-instanceNormScale",
                                            checkpoint+"-conv2-instanceNormShift"),
            ConvolutionLayer(size: ConvSize(layerSize: LayerSize(f: Constants.baseKernelCount * 4, w: Constants.inputImageWidth/4),
                                            kernelSize: 3, stride: 2),
                             weightsFiles:  checkpoint+"-conv3",
                                            checkpoint+"-conv3-instanceNormScale",
                                            checkpoint+"-conv3-instanceNormShift"),
            ResidualLayer(size: LayerSize(f: Constants.baseKernelCount * 4, w: Constants.inputImageWidth/4),
                          weightFiles:  checkpoint+"-res_block_1A",
                                        checkpoint+"-res_block_1A-instanceNormScale",
                                        checkpoint+"-res_block_1A-instanceNormShift",
                                        checkpoint+"-res_block_1B",
                                        checkpoint+"-res_block_1B-instanceNormScale",
                                        checkpoint+"-res_block_1B-instanceNormShift"),
            ResidualLayer(size: LayerSize(f: Constants.baseKernelCount * 4, w: Constants.inputImageWidth/4),
                          weightFiles:  checkpoint+"-res_block_2A",
                                        checkpoint+"-res_block_2A-instanceNormScale",
                                        checkpoint+"-res_block_2A-instanceNormShift",
                                        checkpoint+"-res_block_2B",
                                        checkpoint+"-res_block_2B-instanceNormScale",
                                        checkpoint+"-res_block_2B-instanceNormShift"),
            ResidualLayer(size: LayerSize(f: Constants.baseKernelCount * 4, w: Constants.inputImageWidth/4),
                          weightFiles:  checkpoint+"-res_block_3A",
                                        checkpoint+"-res_block_3A-instanceNormScale",
                                        checkpoint+"-res_block_3A-instanceNormShift",
                                        checkpoint+"-res_block_3B",
                                        checkpoint+"-res_block_3B-instanceNormScale",
                                        checkpoint+"-res_block_3B-instanceNormShift"),
            ResidualLayer(size: LayerSize(f: Constants.baseKernelCount * 4, w: Constants.inputImageWidth/4),
                          weightFiles:  checkpoint+"-res_block_4A",
                                        checkpoint+"-res_block_4A-instanceNormScale",
                                        checkpoint+"-res_block_4A-instanceNormShift",
                                        checkpoint+"-res_block_4B",
                                        checkpoint+"-res_block_4B-instanceNormScale",
                                        checkpoint+"-res_block_4B-instanceNormShift"),
            ResidualLayer(size: LayerSize(f: Constants.baseKernelCount * 4, w: Constants.inputImageWidth/4),
                          weightFiles:  checkpoint+"-res_block_5A",
                                        checkpoint+"-res_block_5A-instanceNormScale",
                                        checkpoint+"-res_block_5A-instanceNormShift",
                                        checkpoint+"-res_block_5B",
                                        checkpoint+"-res_block_5B-instanceNormScale",
                                        checkpoint+"-res_block_5B-instanceNormShift"),
            ConvTransposeLayer(device: device, size: ConvSize(layerSize: LayerSize(f: Constants.baseKernelCount * 2, w: Constants.inputImageWidth/2), kernelSize: 3, stride: 2),
                                    weightsFiles:   checkpoint+"-conv_transp1",
                                                    checkpoint+"-conv_transp1-instanceNormScale",
                                                    checkpoint+"-conv_transp1-instanceNormShift"),
            ConvTransposeLayer(device: device, size: ConvSize(layerSize: LayerSize(f: Constants.baseKernelCount, w: Constants.inputImageWidth), kernelSize: 3, stride: 2),
                                    weightsFiles:   checkpoint+"-conv_transp2",
                                                    checkpoint+"-conv_transp2-instanceNormScale",
                                                    checkpoint+"-conv_transp2-instanceNormShift"),
            ConvolutionLayer(size: ConvSize(layerSize: LayerSize(f: 3, w: Constants.inputImageWidth),
                                                        kernelSize: 9, stride: 1),
                                         neuronType: .tanh,
                                         weightsFiles:  checkpoint+"-convFinal",
                                         checkpoint+"-convFinal-instanceNormScale",
                                         checkpoint+"-convFinal-instanceNormShift"),
            ScaleToFloatLayer(device: device),
            LuminanceLayer(device: device, layerSize: LayerSize(f: 3, w: Constants.inputImageWidth)),
            LanczosLayer(layerSize: LayerSize(f: 3, w: Constants.outputImageWidth)),
            WatermarkLayer(device: device)
            ])
    }
    
    
    public func transform(image inputImage: MPSImage, queue: MTLCommandQueue, result: @escaping (MTLTexture) -> Void) {

        queue.insertDebugCaptureBoundary()
        let buffer = queue.makeCommandBuffer()
        buffer.label = "Network run buffer"
        network.run(inputImage: inputImage, commandBuffer: buffer) { image in
            result(image.texture)
        }
    }

    public func changeToCheckpoint(checkpoint: String) {
        if checkpoint == oldCheckpoint {
            return 
        }
        for layer in network.layers {
            layer.updateCheckpoint(new: checkpoint, old: oldCheckpoint, device: device)
        }
        oldCheckpoint = checkpoint
    }

}
