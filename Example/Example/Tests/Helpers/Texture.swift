//
//  Texture.swift
//  Bender
//
//  Created by Diego Ernst on 5/25/17.
//
//

import Accelerate
import AVFoundation
import MetalKit
import MetalPerformanceShadersProxy
import Bender

public class Texture {

    var data: [[Float]]
    let size: LayerSize

    init(data: [[Float]], size: LayerSize) {
        self.data = data
        self.size = size
    }

    subscript(x: Int, y: Int, z: Int) -> Float {
        get {
            return data[y * width + x][z]
        }
        set {
            data[y * width + x][z] = newValue
        }
    }
    
    var width: Int {
        return size.w
    }
    
    var height: Int {
        return size.h
    }
    
    var depth: Int {
        return size.f
    }

    func isEqual(to texture: Texture, threshold: Float = 0.00001) -> Bool {
        guard self.size == texture.size else { return false }
        for x in 0..<width {
            for y in 0..<height {
                for z in 0..<depth {
                    if abs(self[x, y, z] - texture[x, y, z]) > threshold {
                        return false
                    }
                }
            }
        }
        return true
    }

    var totalCount: Int {
        return size.f * size.w * size.h
    }

}

// MARK: - CustomStringConvertible

extension Texture: CustomStringConvertible {

    public var description: String {
        return data.description
    }

}

// MARK: - Convenience initializer from a metal texture

public extension Texture {

    convenience init(metalTexture: MTLTexture, size: LayerSize) {
        let texture = metalTexture
        let data = [[Float]](repeating: [Float](repeating: 0, count: size.f), count: size.w * size.h)
        let count = texture.width * texture.height * texture.arrayLength * 4
        var bytes = [UInt16](repeating: 0, count: count)
        let region = MTLRegionMake2D(0, 0, size.w, size.h)
        for i in 0..<texture.arrayLength {
            texture.getBytes(
                &bytes + i * 4 * size.w * size.h * MemoryLayout<UInt16>.stride,
                bytesPerRow: size.w * 4 * MemoryLayout<UInt16>.stride,
                bytesPerImage: 0,
                from: region,
                mipmapLevel: 0,
                slice: i
            )
        }
        var output = [Float](repeatElement(0, count: count))
        var bufferFloat16 = vImage_Buffer(data: &bytes, height: 1, width: UInt(count), rowBytes: count * 2)
        var bufferFloat32 = vImage_Buffer(data: &output, height: 1, width: UInt(count), rowBytes: count * 4)
        if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
            fatalError("Error converting float16 to float32")
        }

        self.init(data: data, size: size)

        let sliceSize = size.h * size.w * 4
        for i in 0..<output.count {
            let slice = i / sliceSize
            let index = i - slice*sliceSize
            let z = index % 4 + slice*4
            let y = index / (size.w * 4)
            let x = (index % (size.w * 4)) / 4
            if x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth {
                self[x, y, z] = output[i]
            } // otherwise is a padding value
        }
    }

}

// MARK: - MetalTexture converter

extension Texture {

    public func metalTexture(with device: MTLDevice) -> MTLTexture {
        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: MTLPixelFormat.rgba16Float, width: width, height: height, mipmapped: false)
        desc.textureType = size.f > 4 ? MTLTextureType.type2DArray : MTLTextureType.type2D
        desc.arrayLength = (depth + 3) / 4
        let texture = device.makeTexture(descriptor: desc)
        let region = MTLRegionMake2D(0, 0, width, height)

        for i in 0..<texture.arrayLength {
            let count = width * height * 4
            var data = [Float]()
            for y in 0..<height {
                for x in 0..<width {
                    for z in i*4..<((i+1)*4) {
                        if z >= 0 && z < depth {
                            data.append(self[x, y, z])
                        } else {
                            data.append(0) // padding
                        }
                    }
                }
            }
            var data16 = [UInt16](repeatElement(0, count: count))
            var bufferFloat16 = vImage_Buffer(data: &data16, height: 1, width: UInt(count), rowBytes: count * 2)
            var bufferFloat32 = vImage_Buffer(data: &data, height: 1, width: UInt(count), rowBytes: count * 4)
            if vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0) != kvImageNoError {
                fatalError("Error converting float16 to float32")
            }
            texture.replace(region: region, mipmapLevel: 0, slice: i, withBytes: &data16, bytesPerRow: texture.width * 4 * MemoryLayout<UInt16>.size, bytesPerImage: 0)
        }
        return texture
    }
    
}
