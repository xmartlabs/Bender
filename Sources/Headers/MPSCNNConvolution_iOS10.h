//
//  MPSCNNConvolution_iOS10.h
//  Bender
//
//  Created by Mathias Claassen on 4/3/18.
//

@import MetalPerformanceShaders;

@interface MPSCNNConvolution (MPSCNNConvolution_iOS10)

-(nonnull instancetype) initWithDevice: (nonnull id <MTLDevice>) device
                 convolutionDescriptor: (const MPSCNNConvolutionDescriptor * __nonnull) convolutionDescriptor
                         kernelWeights: (const float * __nonnull) kernelWeights
                             biasTerms: (const float * __nullable) biasTerms
                                 flags: (MPSCNNConvolutionFlags) flags;

@end
