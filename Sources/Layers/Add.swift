//
//  Add.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders

/// Receives two input images. The first is used to take the color and the second is used to take the luminance for the output image.
class Add: NetworkLayerUnion {

    var outputSize: LayerSize!

    // Custom kernels
    let pipelineAdd: MTLComputePipelineState

    // Intermediate images
    var outputImage: MPSImage!

    init(device: MTLDevice) {
        do {
            let library = device.newDefaultLibrary()!
            let kernel = library.makeFunction(name: "sum_matrix")
            pipelineAdd = try device.makeComputePipelineState(function: kernel!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }
    }

    func initialize(device: MTLDevice, prevSize: LayerSize) {
        outputSize = prevSize
        outputImage = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(layerSize: prevSize))
    }

    func updateCheckpoint(new: String, old: String, device: MTLDevice) {}

    func execute(commandBuffer: MTLCommandBuffer, inputImages: MPSImage...) -> MPSImage {
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()
        commandEncoder.label = "sum matrix encoder"
        // mean calculation 1st step
        let tpTG = MTLSizeMake(32, 8, 1)
        commandEncoder.setComputePipelineState(pipelineAdd)

        commandEncoder.setTexture(inputImages[0].texture, at: 0)
        commandEncoder.setTexture(inputImages[1].texture, at: 1)
        commandEncoder.setTexture(outputImage.texture, at: 2)
        let threadgroupsPerGrid = inputImages[0].texture.threadGrid(threadGroup: tpTG)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: tpTG)
        commandEncoder.endEncoding()

        return outputImage
    }
}
