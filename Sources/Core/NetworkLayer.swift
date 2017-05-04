//
//  NetworkLayerProtocol.swift
//  VideoStylizer
//
//  Created by Joaquin Rocco on 12/16/16.
//  Copyright © 2016 Xmartlabs. All rights reserved.
//

import MetalPerformanceShaders
import AVFoundation

protocol NetworkItem {

    var outputSize: LayerSize! { get }
    func initialize(device: MTLDevice, prevSize: LayerSize)
    func execute(commandBuffer: MTLCommandBuffer, inputImage: MPSImage) -> MPSImage
    func updateCheckpoint(new: String, old: String, device: MTLDevice)
    
}

protocol NetworkLayer: NetworkItem {
    
}

protocol NetworkLayerUnion {

    var outputSize: LayerSize! { get }
    func initialize(device: MTLDevice, prevSize: LayerSize)
    func execute(commandBuffer: MTLCommandBuffer, inputImages: MPSImage...) -> MPSImage

}

extension NetworkLayer {

    func loadWeights(from file: String, size: Int, useFloat16: Bool = false) -> UnsafePointer<Float> {
        // Load weights from file(s)
        let typeSize = useFloat16 ? Constants.HalfSize : Constants.FloatSize
        let sizeWeights = size * typeSize

        // get the url to this layer's weights
        let wtPath = Bundle.main.path( forResource: file, ofType: "")

        // open file descriptors in read-only mode to parameter files
        let fd_w  = open( wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)

        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")

        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0);

        // cast Void pointers to Float
        let w = UnsafePointer(hdrW!.bindMemory(to: Float.self, capacity: Int(sizeWeights)))

        close(fd_w)
        assert(w != UnsafePointer<Float>.init(bitPattern: -1), "mmap failed with errno = \(errno)")
        return w
    }

}

//extension NetworkLayer {
//
//    func createImage(device: MTLDevice) -> MPSImage {
//        return MPSImage(device: device, imageDescriptor: descriptor!)
//    }
//
//    func createTempImage(buffer: MTLCommandBuffer, descriptor: MPSImageDescriptor? = nil) -> MPSTemporaryImage {
//        return MPSTemporaryImage(commandBuffer: buffer, imageDescriptor: descriptor ?? self.descriptor!)
//    }
//
//}
