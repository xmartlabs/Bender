//
//  ParameterDataSource.swift
//  MetalBender
//
//  Created by Mathias Claassen on 4/2/18.
//

import MetalPerformanceShaders

@available(iOS 11.0, *)
public class ConvolutionDataSource: NSObject, MPSCNNConvolutionDataSource {

    var useHalf: Bool
    var cnnDescriptor: MPSCNNConvolutionDescriptor
    var weightsPointer: UnsafeMutableRawPointer?
    var biasPointer: UnsafeMutablePointer<Float>?

    var parameterLoader: ParameterLoader?
    var layerId: String = ""
    var weightCount: Int = -1, biasCount: Int = -1

    public init(cnnDescriptor: MPSCNNConvolutionDescriptor, weights: UnsafeMutableRawPointer?,
                bias: UnsafeMutablePointer<Float>?, useHalf: Bool = false) {
        self.useHalf = useHalf
        self.cnnDescriptor = cnnDescriptor
        self.weightsPointer = weights
        self.biasPointer = bias
    }

    public init(cnnDescriptor: MPSCNNConvolutionDescriptor, parameterLoader: ParameterLoader, layerId: String,
                weightCount: Int, biasCount: Int, useHalf: Bool = false) {
        self.useHalf = useHalf
        self.cnnDescriptor = cnnDescriptor
        self.layerId = layerId
        self.weightCount = weightCount
        self.biasCount = biasCount
        self.parameterLoader = parameterLoader
    }

    public func dataType() -> MPSDataType {
        return useHalf ? .float16 : .float32
    }

    public func weights() -> UnsafeMutableRawPointer {
        return weightsPointer!
    }

    public func descriptor() -> MPSCNNConvolutionDescriptor {
        return cnnDescriptor
    }

    public func biasTerms() -> UnsafeMutablePointer<Float>? {
        guard let bias = biasPointer else {
            return nil
        }
        return bias
    }

    public func load() -> Bool {
        if let parameterLoader = parameterLoader {
            biasPointer = UnsafeMutablePointer(mutating: parameterLoader.loadWeights(for: layerId,
                                                                                     modifier: Convolution.biasModifier,
                                                                                     size: biasCount))
            weightsPointer = UnsafeMutableRawPointer(mutating: parameterLoader.loadWeights(for: layerId,
                                                                                           modifier: Convolution.weightModifier,
                                                                                           size: weightCount))
        }
        return true
    }

    public func purge() {
        if parameterLoader != nil {
            biasPointer = nil
            weightsPointer = nil
        }
    }

    public func label() -> String? {
        return nil
    }

}
